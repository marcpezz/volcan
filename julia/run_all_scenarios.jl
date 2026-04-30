# Run one representative scenario per type (S1..S10) in a SINGLE Julia
# session. MTK / GeoParams / ParallelStencil are loaded and JIT-compiled
# only once, then `run_scenario_with_ML_data` is called 10 times in a row.
#
# Usage:
#   julia --project=. julia/run_all_scenarios.jl                 # all 10, frame_interval=20
#   julia --project=. julia/run_all_scenarios.jl 100             # all 10, frame_interval=100
#   julia --project=. julia/run_all_scenarios.jl 20 S3 S7        # only S3 and S7
#   SMOKE=true julia --project=. julia/run_all_scenarios.jl      # quick pipeline check (frame_interval=200)
#
# Outputs:
#   results/ML_data_<global_id>_<scenario>.h5    (one HDF5 per scenario)
#   results/test_summary.txt                     (human-readable table)
#   results/test_summary.csv                     (machine-readable)

using Printf
using Dates

# Load run_animation.jl; the trailing CLI block there is guarded with
# `if abspath(PROGRAM_FILE) == @__FILE__`, so only the function definition
# (plus backend init) runs here.
const _THIS_DIR = @__DIR__
include(joinpath(_THIS_DIR, "run_animation.jl"))

const SCENARIOS_DIR = joinpath(_THIS_DIR, "..", "scenarios")

# First JSON of each scenario group (deterministic).
const SCENARIOS = [
    ("S1",  "scenario_0000_S1.json"),
    ("S2",  "scenario_0100_S2.json"),
    ("S3",  "scenario_0200_S3.json"),
    ("S4",  "scenario_0300_S4.json"),
    ("S5",  "scenario_0400_S5.json"),
    ("S6",  "scenario_0500_S6.json"),
    ("S7",  "scenario_0600_S7.json"),
    ("S8",  "scenario_0700_S8.json"),
    ("S9",  "scenario_0800_S9.json"),
    ("S10", "scenario_0900_S10.json"),
]

function parse_cli()
    frame_interval = 20
    wanted_ids = String[]
    if length(ARGS) >= 1
        try
            frame_interval = parse(Int, ARGS[1])
        catch
            # first arg was not an integer -> treat it as a scenario ID
            push!(wanted_ids, ARGS[1])
        end
    end
    for a in ARGS[2:end]
        push!(wanted_ids, a)
    end

    # SMOKE mode: force a very coarse snapshot cadence for a quick pipeline check
    if get(ENV, "SMOKE", "false") in ("1", "true", "True", "TRUE")
        frame_interval = 200
        @warn "SMOKE=true -> frame_interval forced to $frame_interval (pipeline check only, not scientifically meaningful)"
    end

    return frame_interval, wanted_ids
end

function main()
    frame_interval, wanted_ids = parse_cli()

    # Filter scenario list if the user specified a subset.
    selected = isempty(wanted_ids) ?
        SCENARIOS :
        filter(s -> s[1] in wanted_ids, SCENARIOS)
    if isempty(selected)
        error("No scenarios matched. Requested: $wanted_ids. Available: $(first.(SCENARIOS)).")
    end

    # Pre-flight: make sure every JSON exists before we start anything.
    missing_files = String[]
    for (_, fname) in selected
        p = joinpath(SCENARIOS_DIR, fname)
        isfile(p) || push!(missing_files, p)
    end
    isempty(missing_files) || error("Missing scenario JSONs:\n  " * join(missing_files, "\n  "))

    mkpath("results")

    println("\n" * "#"^70)
    println("# Multi-scenario test run  ($(length(selected)) scenario(s))")
    println("# frame_interval = $frame_interval")
    println("# started at      $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println("#"^70 * "\n")

    # Collect results: (scenario_id, status, h5_file, elapsed_s, error_msg)
    results = Vector{NamedTuple{(:id, :status, :h5, :elapsed_s, :error),
                                Tuple{String,Symbol,String,Float64,String}}}()

    t_suite = Base.time()
    for (i, (sid, fname)) in enumerate(selected)
        json_path = joinpath(SCENARIOS_DIR, fname)
        println("\n" * "="^70)
        println("[$i/$(length(selected))]  $sid   $fname")
        println("="^70)

        try
            h5_file, elapsed_s = run_scenario_with_ML_data(json_path;
                                                          frame_interval = frame_interval)
            push!(results, (id=sid, status=:ok, h5=h5_file,
                            elapsed_s=elapsed_s, error=""))
        catch e
            bt = sprint(showerror, e, catch_backtrace())
            @error "Scenario $sid ($fname) failed" exception=(e, catch_backtrace())
            push!(results, (id=sid, status=:error, h5="-",
                            elapsed_s=0.0, error=first(split(bt, '\n'))))
        end
    end
    suite_elapsed = Base.time() - t_suite

    # ---- Summary ------------------------------------------------------------
    header = @sprintf("%-6s %-8s %-12s  %s", "ID", "STATUS", "RUNTIME[s]", "OUTPUT / ERROR")
    sep    = "-"^max(length(header), 70)

    println("\n" * "#"^70)
    println("# Summary  (total suite runtime: $(round(suite_elapsed, digits=1)) s / $(round(suite_elapsed/60, digits=2)) min)")
    println("#"^70)
    println(header)
    println(sep)
    for r in results
        tail = r.status == :ok ? r.h5 : r.error
        println(@sprintf("%-6s %-8s %-12.1f  %s",
                         r.id, String(r.status), r.elapsed_s, tail))
    end

    n_ok  = count(r -> r.status == :ok,    results)
    n_err = count(r -> r.status == :error, results)
    println(sep)
    println(@sprintf("Ok: %d   Failed: %d   Total: %d", n_ok, n_err, length(results)))

    # ---- Persist summary ----------------------------------------------------
    txt_path = joinpath("results", "test_summary.txt")
    csv_path = joinpath("results", "test_summary.csv")

    open(txt_path, "w") do io
        println(io, "# Multi-scenario test run")
        println(io, "# Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "# frame_interval = $frame_interval")
        println(io, "# suite_runtime_s = $(round(suite_elapsed, digits=1))")
        println(io, header)
        println(io, sep)
        for r in results
            tail = r.status == :ok ? r.h5 : r.error
            println(io, @sprintf("%-6s %-8s %-12.1f  %s",
                                 r.id, String(r.status), r.elapsed_s, tail))
        end
        println(io, sep)
        println(io, @sprintf("Ok: %d   Failed: %d   Total: %d", n_ok, n_err, length(results)))
    end

    open(csv_path, "w") do io
        println(io, "scenario_id,status,runtime_s,h5_file,error")
        for r in results
            # crude CSV escaping: wrap error in quotes if it has a comma
            err = replace(r.error, "\"" => "'")
            if occursin(',', err); err = "\"" * err * "\""; end
            println(io, @sprintf("%s,%s,%.3f,%s,%s",
                                 r.id, String(r.status), r.elapsed_s, r.h5, err))
        end
    end

    println("\n✓ Summary written to:")
    println("    $txt_path")
    println("    $csv_path")

    # Non-zero exit code if anything failed, useful for CI / shell wrappers.
    if n_err > 0
        exit(1)
    end
end

# Same entry-point guard as run_animation.jl: only run when launched directly.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
