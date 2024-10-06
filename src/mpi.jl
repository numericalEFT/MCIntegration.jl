abstract type AbstractParallelMode end
struct NoParallel <: AbstractParallelMode end
struct MPIMode <: AbstractParallelMode end
struct ThreadMode <: AbstractParallelMode end

abstract type AbstractSolver end
struct MCMC <: AbstractSolver end
struct VegasMC <: AbstractSolver end

function mpi_run(f, tasks)
    ########### initialized MPI #######################################
    (MPI.Initialized() == false) && MPI.Init()
    comm = MPI.COMM_WORLD
    nblock = MPI.Comm_size(comm)
    ntasks_each_block = length(tasks) ÷ nblock
    rank = MPI.Comm_rank(comm)
    for i in eachindex(tasks)
        if i % ntasks_each_block == rank
            f(tasks[i])
        end
    end
    # collect data
    data = MPI.Gather(rank == 0 ? MPI.ROOT : MPI.PROC_NULL, tasks, 0, comm)
    return data
end

struct MCTask
    seed::Int
    parameters::Vector{Float64}
end

function single_step!(::MPIMode, solver, integrand, config, task)
    results = mpi_run((task) -> run_single_task(solver, integrand, config, task), config.tasks)
    comm = MPI.COMM_WORLD
    if MPI.Comm_rank(comm) == 0
        analysed = analyze_results(results)
    end
    # scatter adaptive parameters to other workers
    MPI.Bcast(analysed, 0, comm)
end

function integrator(mode::NoParallel, solver::AbstractSolver, integrand::Function, config::Config)
    ...
end
function integrator(mode::MPIMode, solver::AbstractSolver, integrand::Function, config::Config, nstasks::Int)
    results = []
    tasks = Random.rand(Int, nstasks)
    params = tasks.parameters
    for iter in 1:niter
        res = single_step!(mode, solver, integrand, config, tasks[iter])
        params = res.parameters
        push!(results, res)
    end
    # analyse results
end