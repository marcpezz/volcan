# Run simulation with HDF5 data export for ML post-processing
# Saves comprehensive snapshots: temperature, melt fraction, phases, metadata
# GIF/animation generation removed
# Reports total runtime at the end

using JSON
using ParallelStencil
using MagmaThermoKinematics
using MagmaThermoKinematics: kyr, Myr, km, km³, SecYear
using GeoParams
using HDF5
using Statistics

# ---------------------------------------------------------------
# Backend selection (CPU / GPU).
# Toggle with the USE_GPU environment variable, e.g.:
#     USE_GPU=true julia --project=. run_animation.jl scenario.json
# or edit the default below.
# MTK only supports CUDA (NVIDIA) GPUs via CUDA.jl.
# ---------------------------------------------------------------
const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : false

using MagmaThermoKinematics: environment!
if USE_GPU
    using CUDA
    CUDA.functional() || error("USE_GPU=true but CUDA is not functional on this machine")
    environment!(:gpu, Float64, 2)
else
    environment!(:cpu, Float64, 2)
end

# `environment!` initialises ParallelStencil *inside the MTK submodules*, but
# the @parallel macros used at top level of this file (e.g. GridArray!,
# assign!) live in Main, which needs its own @init_parallel_stencil with the
# matching backend.
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using ParallelStencil.FiniteDifferences2D
using MagmaThermoKinematics.Diffusion2D
using MagmaThermoKinematics.Fields2D
using MagmaThermoKinematics.MTK_GMG_2D
using MagmaThermoKinematics.MTK_GMG

function run_scenario_with_ML_data(json_file::String; frame_interval::Int=20)

    t_start = Base.time()

    println("="^60)
    println("MTK Simulation: HDF5 Data Export")
    println("Backend: ", USE_GPU ? "GPU (CUDA)" : "CPU (Threads)")
    println("="^60)

    params = JSON.parsefile(json_file)
    scenario_id = params["scenario_id"]
    scenario_name = params["scenario_name"]

    println("Scenario: $scenario_name ($scenario_id)")
    println("Saving data every $frame_interval timesteps")

    Nx, Nz = params["grid_nx"], params["grid_nz"]
    W_km, H_km = params["domain_width_km"], params["domain_depth_km"]
    Grid = CreateGrid(size=(Nx, Nz), extent=(W_km * 1e3, H_km * 1e3))
    Num = Numeric_params(verbose=false)

    k_host = params["host_conductivity"]
    cp_host = params["host_heat_capacity"]

    MatParam = (
        SetMaterialParams(
            Name = "Host rock", Phase = 1,
            Density = ConstantDensity(ρ=2700kg/m^3),
            HeatCapacity = ConstantHeatCapacity(Cp=cp_host * J / kg / K),
            Conductivity = ConstantConductivity(k=k_host * Watt / K / m),
            LatentHeat = ConstantLatentHeat(Q_L=0J/kg)
        ),
        SetMaterialParams(
            Name = "Intruded magma", Phase = 2,
            Density = ConstantDensity(ρ=2700kg/m^3),
            HeatCapacity = T_HeatCapacity_Whittington(),
            Conductivity = T_Conductivity_Whittington(),
            LatentHeat = ConstantLatentHeat(Q_L=2.67e5J/kg),
            Melting = SmoothMelting(MeltingParam_Quadratic(
                T_s = (700 + 273.15)K,
                T_l = (1100 + 273.15)K
            ))
        )
    )

    GeoT = params["geotherm_gradient"] / 1e3

    injection_pattern = get(params, "injection_pattern", "steady")
    if injection_pattern == "radial_accretion"
        W_in = get(params, "sill_diameter_km", params["intrusion_width"] / 1e3) * 1e3
        H_in = get(params, "sill_thickness_m", params["dike_thickness"])
    elseif injection_pattern == "elliptical_sills"
        semi_major = get(params, "sill_semi_major_km", 3.0) * 1e3
        aspect_ratio = get(params, "sill_aspect_ratio", 4.0)
        semi_minor = semi_major / aspect_ratio
        W_in = 2.0 * semi_major
        H_in = 2.0 * semi_minor
    else
        W_in = params["intrusion_width"]
        H_in = params["dike_thickness"]
    end

    T_in = params["magma_temp_C"]
    InjectionInterval = params["injection_interval_yr"] * 3600 * 24 * 365.25
    maxTime = params["simulation_duration_Myr"] * 1e6 * 3600 * 24 * 365.25

    if scenario_id == "S4"
        depth_center = -params["depth_deep_km"] * 1e3
    elseif scenario_id == "S5"
        depth_center = -params["depth_stage1_km"] * 1e3
    elseif haskey(params, "depth_center_km")
        depth_center = -params["depth_center_km"] * 1e3
    else
        depth_center = -15e3
    end

    κ = k_host / (2700 * cp_host)
    dt = minimum(Grid.Δ .^ 2) / κ / 10
    nt = floor(Int64, maxTime / dt)
    nTr_dike = 500

    println("Grid: $Nx x $Nz, dt: $(round(dt / (3600 * 24 * 365.25), digits=1)) yr")
    println("Expected snapshots: ~$(div(nt, frame_interval))")

    Arrays = CreateArrays(Dict(
        (Nx, Nz) => (
            T = 0, T_K = 0, T_it_old = 0, K = k_host, Rho = 2700, Cp = cp_host,
            Tnew = 0, Hr = 0, Hl = 0, Kc = 1, P = 0, X = 0, Z = 0, ϕₒ = 0, ϕ = 0, dϕdT = 0
        ),
        (Nx - 1, Nz) => (qx = 0, Kx = 0),
        (Nx, Nz - 1) => (qz = 0, Kz = 0)
    ))

    Tnew_cpu = Matrix{Float64}(undef, Grid.N...)
    Phi_melt_cpu = similar(Tnew_cpu)
    # `Phases_cpu` is the host buffer written by `PhasesFromTracers!` (scalar
    # indexing on the host). `Phases` is what is passed to GPU kernels: on GPU
    # it must live in device memory; on CPU they share the same array.
    Phases_cpu = ones(Int64, Grid.N...)
    Phases     = USE_GPU ? Data.Array(Phases_cpu) : Phases_cpu

    @parallel (1:Nx, 1:Nz) GridArray!(Arrays.X, Arrays.Z, Grid.coord1D[1], Grid.coord1D[2])

    Tracers = StructArray{Tracer}(undef, 1)

    injection_pattern = get(params, "injection_pattern", "steady")

    if injection_pattern == "radial_accretion"
        x_center = 0.0
        dike_type = "EllipticalIntrusion"
        dike_angle = [0.0, 0.0, 0.0]
    elseif injection_pattern == "elliptical_sills"
        x_center = 0.0
        dike_type = "EllipticalIntrusion"
        dike_angle = [0.0, 0.0, 0.0]
    else
        x_center = W_km * 1e3 / 2.0
        dike_type = "ElasticDike"
        dike_angle = [0.0, 0.0, 0.0]
    end

    dike = Dike(W = W_in, H = H_in, Type = dike_type, T = T_in, Center = [x_center, depth_center], Angle = dike_angle)
    Arrays.T .= -Arrays.Z .* GeoT

    sim_time, dike_inj, InjectVol = 0.0, 0.0, 0.0
    h5_file = "results/ML_data_$(params["global_id"])_$(scenario_id).h5"
    mkpath("results")
    snapshot_count = 0

    println("\nRunning simulation...")

    for it = 1:nt
        if floor(sim_time / InjectionInterval) > dike_inj
            dike_inj = floor(sim_time / InjectionInterval)
            Tnew_cpu .= Array(Arrays.T)

            current_depth = depth_center
            if injection_pattern == "two_stage"
                time_Myr = sim_time / (1e6 * 3600 * 24 * 365.25)
                transition_time = params["transition_time_Myr"]
                if time_Myr >= transition_time
                    current_depth = -params["depth_stage2_km"] * 1e3
                else
                    current_depth = -params["depth_stage1_km"] * 1e3
                end
            elseif injection_pattern == "transcrustal"
                depth_shallow = -params["depth_shallow_km"] * 1e3
                depth_deep = -params["depth_deep_km"] * 1e3
                current_depth = depth_shallow + rand() * (depth_deep - depth_shallow)
            end

            inject_now = true
            if injection_pattern == "episodic_pulses"
                time_kyr = sim_time / (1e3 * 3600 * 24 * 365.25)
                active_dur = params["active_duration_kyr"]
                quiet_dur = params["quiet_duration_kyr"]
                cycle_dur = active_dur + quiet_dur
                time_in_cycle = mod(time_kyr, cycle_dur)
                if time_in_cycle > active_dur
                    inject_now = false
                end
            end

            if !inject_now
                continue
            end

            if injection_pattern == "radial_accretion"
                x_inject = 0.0
                z_inject = depth_center
                dike = Dike(W = W_in, H = H_in, Type = dike_type, T = T_in,
                            Center = [x_inject, z_inject], Angle = dike_angle)
            elseif injection_pattern == "elliptical_sills"
                x_inject = 0.0
                z_inject = current_depth
                dike = Dike(W = W_in, H = H_in, Type = dike_type, T = T_in,
                            Center = [x_inject, z_inject], Angle = dike_angle)
            else
                random_angle_deg = -60.0 + rand() * 120.0
                random_x_offset = (rand() - 0.5) * 10e3
                x_inject = x_center + random_x_offset

                if injection_pattern == "transcrustal"
                    z_inject = current_depth
                else
                    random_z_offset = (rand() - 0.5) * 4e3
                    z_inject = current_depth + random_z_offset
                end

                dike = Dike(
                    W = W_in, H = H_in, Type = dike_type, T = T_in,
                    Center = [x_inject, z_inject], Angle = [random_angle_deg, 0.0, 0.0]
                )
            end

            Tracers, Tnew_cpu, Vol = InjectDike(Tracers, Tnew_cpu, Grid.coord1D, dike, nTr_dike)
            Arrays.T .= Data.Array(Tnew_cpu)
            InjectVol += Vol
            # PhasesFromTracers! does scalar writes -> must run on a host array.
            # On GPU we update Phases_cpu and then upload it to the device buffer.
            PhasesFromTracers!(Phases_cpu, Grid, Tracers, BackgroundPhase = 1, InterpolationMethod = "Constant")
            if USE_GPU
                copyto!(Phases, Phases_cpu)
            end
            println("  Injection #$(Int(dike_inj)) at t=$(round(sim_time / (3600 * 24 * 365.25 * 1e3), digits=1)) kyr")
        end

        Nonlinear_Diffusion_step_2D!(Arrays, MatParam, Phases, Grid, dt, Num)
        copy_arrays_GPU2CPU!(Tnew_cpu, Phi_melt_cpu, Arrays.Tnew, Arrays.ϕ)
        UpdateTracers_T_ϕ!(Tracers, Grid.coord1D, Tnew_cpu, Phi_melt_cpu)
        @parallel assign!(Arrays.T, Arrays.Tnew)
        @parallel assign!(Arrays.Tnew, Arrays.T)
        sim_time = sim_time + dt

        if mod(it, frame_interval) == 0
            snapshot_count += 1

            file = h5open(h5_file, snapshot_count == 1 ? "w" : "r+")
            try
                if snapshot_count == 1
                    g = create_group(file, "grid")
                    g["x"] = collect(Grid.coord1D[1])
                    g["z"] = collect(Grid.coord1D[2])
                    g["nx"] = Nx
                    g["nz"] = Nz

                    p = create_group(file, "parameters")
                    for (k, v) in params
                        if v isa Number
                            p[k] = v
                        elseif v isa String
                            p[k] = v
                        end
                    end

                    create_group(file, "snapshots")
                end

                s = create_group(file["snapshots"], "snap_$(snapshot_count - 1)")
                s["timestep"] = it
                s["time_kyr"] = sim_time / (3600 * 24 * 365.25 * 1e3)
                s["injection_count"] = Int(dike_inj)
                s["volume_km3"] = InjectVol / 1e9

                T_field = Array(Arrays.T)
                phi_field = Array(Arrays.ϕ)
                s["temperature"] = permutedims(T_field)
                s["melt_fraction"] = permutedims(phi_field)
                s["phases"] = permutedims(Phases_cpu)

                s["heat_flux_x"] = permutedims(Array(Arrays.qx))
                s["heat_flux_z"] = permutedims(Array(Arrays.qz))

                s["conductivity"] = permutedims(Array(Arrays.K))
                s["heat_capacity"] = permutedims(Array(Arrays.Cp))
                s["latent_heat"] = permutedims(Array(Arrays.Hl))

                s["T_max"] = maximum(T_field)
                s["T_mean"] = sum(T_field) / prod(Grid.N)
                s["T_min"] = minimum(T_field)
                s["T_std"] = std(T_field)
                s["phi_max"] = maximum(phi_field)
                s["phi_mean"] = sum(phi_field) / prod(Grid.N)
                s["phi_volume_fraction"] = sum(phi_field) / prod(Grid.N)

                s["heat_flux_x_max"] = maximum(abs.(Array(Arrays.qx)))
                s["heat_flux_z_max"] = maximum(abs.(Array(Arrays.qz)))
                s["total_heat_flux"] = sqrt(sum(Array(Arrays.qx) .^ 2) + sum(Array(Arrays.qz) .^ 2))

                s["magma_fraction"] = sum(Phases_cpu .== 2) / prod(Grid.N)
                s["host_rock_fraction"] = sum(Phases_cpu .== 1) / prod(Grid.N)

                eruptible_threshold_low = 0.4
                eruptible_threshold_high = 0.5

                eruptible_cells_low = sum(phi_field .>= eruptible_threshold_low)
                eruptible_cells_high = sum(phi_field .>= eruptible_threshold_high)
                cell_volume = (Grid.coord1D[1][2] - Grid.coord1D[1][1]) * (Grid.coord1D[2][2] - Grid.coord1D[2][1])

                s["eruptible_volume_phi40_km3"] = eruptible_cells_low * cell_volume / 1e9
                s["eruptible_volume_phi50_km3"] = eruptible_cells_high * cell_volume / 1e9
                s["eruptible_fraction_phi40"] = eruptible_cells_low / prod(Grid.N)
                s["eruptible_fraction_phi50"] = eruptible_cells_high / prod(Grid.N)

                eruptible_mask_low = phi_field .>= eruptible_threshold_low
                if any(eruptible_mask_low)
                    z_coords = Grid.coord1D[2]
                    eruptible_depths = [z_coords[j] for i in 1:Nx, j in 1:Nz if eruptible_mask_low[i, j]]
                    s["min_eruptible_depth_km"] = maximum(eruptible_depths) / 1e3
                    s["max_eruptible_depth_km"] = minimum(eruptible_depths) / 1e3
                    s["avg_melt_in_eruptible_zone"] = sum(phi_field[eruptible_mask_low]) / eruptible_cells_low
                else
                    s["min_eruptible_depth_km"] = 0.0
                    s["max_eruptible_depth_km"] = 0.0
                    s["avg_melt_in_eruptible_zone"] = 0.0
                end

                if eruptible_cells_low > 0
                    depth_factor = exp(maximum(eruptible_depths) / 5e3)
                    volume_factor = min(1.0, eruptible_cells_low * cell_volume / 1e8)
                    melt_factor = sum(phi_field[eruptible_mask_low]) / eruptible_cells_low
                    s["eruptibility_index"] = depth_factor * volume_factor * melt_factor
                else
                    s["eruptibility_index"] = 0.0
                end

                if length(Tracers) > 0 && isassigned(Tracers, 1)
                    t = create_group(s, "tracers")

                    n_tracers = length(Tracers)
                    t["n_tracers"] = n_tracers

                    tracer_x = zeros(n_tracers)
                    tracer_z = zeros(n_tracers)
                    tracer_T = zeros(n_tracers)
                    tracer_phase = zeros(Int64, n_tracers)
                    tracer_phi = zeros(n_tracers)
                    tracer_id = zeros(Int64, n_tracers)

                    for (i, tr) in enumerate(Tracers)
                        tracer_x[i] = tr.coord[1]
                        tracer_z[i] = tr.coord[2]
                        tracer_T[i] = tr.T
                        tracer_phase[i] = tr.Phase
                        tracer_phi[i] = tr.Phi
                        tracer_id[i] = tr.num
                    end

                    t["x"] = tracer_x
                    t["z"] = tracer_z
                    t["temperature"] = tracer_T
                    t["phase"] = tracer_phase
                    t["melt_fraction"] = tracer_phi
                    t["tracer_id"] = tracer_id

                    magma_tracers = tracer_phase .== 2
                    if any(magma_tracers)
                        t["n_magma_tracers"] = sum(magma_tracers)
                        t["magma_T_mean"] = mean(tracer_T[magma_tracers])
                        t["magma_T_max"] = maximum(tracer_T[magma_tracers])
                        t["magma_T_min"] = minimum(tracer_T[magma_tracers])
                        t["magma_phi_mean"] = mean(tracer_phi[magma_tracers])
                        t["magma_depth_mean"] = mean(tracer_z[magma_tracers]) / 1e3
                        t["magma_depth_min"] = maximum(tracer_z[magma_tracers]) / 1e3
                        t["magma_depth_max"] = minimum(tracer_z[magma_tracers]) / 1e3
                    else
                        t["n_magma_tracers"] = 0
                        t["magma_T_mean"] = 0.0
                        t["magma_T_max"] = 0.0
                        t["magma_T_min"] = 0.0
                        t["magma_phi_mean"] = 0.0
                        t["magma_depth_mean"] = 0.0
                        t["magma_depth_min"] = 0.0
                        t["magma_depth_max"] = 0.0
                    end
                end
            finally
                close(file)
            end

            mod(snapshot_count, 10) == 0 && println("  Snapshot $snapshot_count saved")
        end
    end

    elapsed_s = Base.time() - t_start
    elapsed_min = elapsed_s / 60.0

    file = h5open(h5_file, "r+")
    try
        file["runtime_seconds"] = elapsed_s
        file["runtime_minutes"] = elapsed_min
    finally
        close(file)
    end

    println("\n✓ ML Data: $h5_file ($snapshot_count snapshots)")
    println("  Injections: $(Int(dike_inj)), Volume: $(round(InjectVol / 1e9, digits=2)) km³")
    println("  Final T_max: $(round(maximum(Array(Arrays.T)), digits=1))°C, φ_max: $(round(maximum(Array(Arrays.ϕ)), digits=3))")
    println("  Total runtime: $(round(elapsed_s, digits=1)) s ($(round(elapsed_min, digits=2)) min)")

    return h5_file, elapsed_s
end

# Only execute the CLI entry point when this file is launched directly
# (`julia run_animation.jl scenario.json`). When another script `include`s it
# — e.g. run_all_scenarios.jl — we just want the function definition, not
# the side effects of reading ARGS and launching a simulation.
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia --project=. run_animation.jl <scenario.json> [frame_interval]")
        exit(1)
    end

    h5_file, elapsed_s = run_scenario_with_ML_data(
        ARGS[1],
        frame_interval = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 20
    )

    println("\n✓ Complete! Saved to: $h5_file")
    println("✓ Runtime: $(round(elapsed_s, digits=1)) s")
end