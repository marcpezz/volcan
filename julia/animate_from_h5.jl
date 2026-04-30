# animate_from_h5.jl
#
# Usage:
#   julia --project=. animate_from_h5.jl results/ML_data_0_S1.h5 [fps]
#
# Notes:
# - Expects the HDF5 structure produced by your ML exporter:
#     /grid/x
#     /grid/z
#     /snapshots/snap_0/temperature
#     /snapshots/snap_0/melt_fraction
#     /snapshots/snap_0/time_kyr
#     ...
# - Writes a GIF next to the input file unless an output path is set below.

using HDF5
using Plots
using Printf

function sorted_snapshot_names(g_snaps)
    names = String.(collect(keys(g_snaps)))
    names = filter(n -> startswith(n, "snap_"), names)
    sort(names, by = n -> parse(Int, split(n, "_")[end]))
end

function load_snapshots(h5_file::String)
    x = Float64[]
    z = Float64[]
    times_kyr = Float64[]
    T_frames = Matrix{Float64}[]
    ϕ_frames = Matrix{Float64}[]

    h5open(h5_file, "r") do file
        haskey(file, "grid") || error("Missing /grid group in $h5_file")
        haskey(file, "snapshots") || error("Missing /snapshots group in $h5_file")

        x = read(file["grid/x"]) ./ 1e3
        z = read(file["grid/z"]) ./ 1e3

        g_snaps = file["snapshots"]
        snap_names = sorted_snapshot_names(g_snaps)
        isempty(snap_names) && error("No snap_i groups found in /snapshots")

        for name in snap_names
            g = g_snaps[name]

            haskey(g, "temperature") || error("Missing temperature in /snapshots/$name")
            haskey(g, "melt_fraction") || error("Missing melt_fraction in /snapshots/$name")

            push!(T_frames, Matrix{Float64}(read(g["temperature"])))
            push!(ϕ_frames, Matrix{Float64}(read(g["melt_fraction"])))

            if haskey(g, "time_kyr")
                push!(times_kyr, Float64(read(g["time_kyr"])))
            elseif haskey(g, "time_s")
                push!(times_kyr, Float64(read(g["time_s"])) / (3600 * 24 * 365.25 * 1e3))
            else
                push!(times_kyr, length(times_kyr))
            end
        end
    end

    return x, z, times_kyr, T_frames, ϕ_frames
end

function animate_h5(
    h5_file::String;
    fps::Int = 10,
    output_file::Union{Nothing,String} = nothing,
    T_clims::Tuple{Float64,Float64} = (0.0, 1200.0),
    ϕ_clims::Tuple{Float64,Float64} = (0.0, 1.0),
)
    x, z, times_kyr, T_frames, ϕ_frames = load_snapshots(h5_file)

    if output_file === nothing
        stem = splitext(basename(h5_file))[1]
        output_file = joinpath(dirname(h5_file), stem * "_recreated.gif")
    end

    ENV["GKSwstype"] = "nul"

    anim = @animate for i in eachindex(times_kyr)
        p1 = heatmap(
            x, z, T_frames[i];
            aspect_ratio = 1,
            c = :hot,
            clims = T_clims,
            xlabel = "Width [km]",
            ylabel = "Depth [km]",
            title = @sprintf("%.1f kyr", times_kyr[i]),
            dpi = 150,
            fontsize = 8,
            colorbar_title = "T [°C]"
        )

        p2 = heatmap(
            x, z, ϕ_frames[i];
            aspect_ratio = 1,
            c = :YlOrRd,
            clims = ϕ_clims,
            xlabel = "Width [km]",
            title = "Melt Fraction",
            dpi = 150,
            fontsize = 8,
            colorbar_title = "φ"
        )

        plot(p1, p2; layout = (1, 2), size = (1000, 400))
    end

    gif(anim, output_file; fps = fps)
    println("Saved animation to: $output_file")
    return output_file
end

function main()
    if length(ARGS) < 1
        println("Usage: julia --project=. animate_from_h5.jl <ML_data.h5> [fps]")
        exit(1)
    end

    h5_file = ARGS[1]
    fps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10

    animate_h5(h5_file; fps = fps)
end

main()