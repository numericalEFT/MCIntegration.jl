"""
    function integrate(integrand::Function;
        solver::Symbol=:vegas, # :mcmc, :vegas, or :vegasmc
        config::Union{Configuration,Nothing}=nothing,
        neval=1e4, 
        niter=10, 
        block=16, 
        print=-1, 
        gamma=1.0, 
        adapt=true,
        debug=false, 
        reweight_goal::Union{Vector{Float64},Nothing}=nothing, 
        ignore::Int=adapt ? 1 : 0,
        measure::Union{Nothing,Function}=nothing,
        measurefreq::Int=1,
        inplace::Bool=false,
        kwargs...
    )

 Calculate the integrals, collect statistics, and return a Result struct that contains the estimations and errors.

 # Remarks
 - User may run the MC in parallel using MPI. Simply run `mpiexec -n N julia userscript.jl` where `N` is the number of workers. In this mode, only the root process returns meaningful results. All other workers return `nothing, nothing`. User is responsible to handle the returning results properly. If you have multiple number of mpi version, you can use "mpiexecjl" in your "~/.julia/package/MPI/###/bin" to make sure the version is correct. See https://juliaparallel.github.io/MPI.jl/stable/configuration/ for more detail.
 - In the MC, a normalization diagram is introduced to normalize the MC estimates of the integrands. More information can be found in the link: https://kunyuan.github.io/QuantumStatistics.jl/dev/man/important_sampling/#Important-Sampling. User don't need to explicitly specify this normalization diagram.Internally, normalization diagram will be added to each table that is related to the integrands.

 # Arguments

- `integrand`:Function call to evaluate the integrand.  
              If `inplace = false`, then the signature of the integrand is `integrand(var, config)`, where `var` is a vector of random variables, and `config` is the [`Configuration`](@ref) struct. It should return one or more weights, corresponding to the value of each component of the integrand for the given `var`.
              If `inplace = true``, then the signature of the integrand is `integrand(var, weights, config)`, where the additional argument `weights` is the value of the integrand components for the given `var`.
              Internally, MC only samples the absolute value of the weight. Therefore, it is also important to define Main.abs for the weight if its type is user-defined. 
- `solver` :  :vegas, :vegasmc, or :mcmc. See Readme for more details.
- `config`:   [`Configuration`](@ref) object to perform the MC integration. If `nothing`, it attempts to create a new one with Configuration(; kwargs...).
- `neval`:    Number of evaluations of the integrand per iteration. 
- `niter`:    Number of iterations. The reweight factor and the variables will be self-adapted after each iteration. 
- `block`:    Number of blocks. Each block will be evaluated by about neval/block times. Each block is assumed to be statistically independent, and will be used to estimate the error. 
              In MPI mode, the blocks are distributed among the workers. If the numebr of workers N is larger than block, then block will be set to be N.
- `print`:    -2 to not print anything; -1 to print minimal information; 0 to print the iteration history in the end; >0 to print MC configuration for every `print` seconds and print the iteration history in the end.
- `gamma`:    Learning rate of the reweight factor after each iteraction. Note that alpha <=1, where alpha = 0 means no reweighting.  
- `adapt`:    Whether to adapt the grid and the reweight factor.
- `debug`:    Whether to print debug information (type instability, float overflow etc.)
- `reweight_goal`: The expected distribution of visited times for each integrand after reweighting . If not set, then all factors will be initialized with one. Only useful for the :mcmc solver. 
- `ignore`:   ignore the iteration until the `ignore` round. By default, the first iteration is igonred if adapt=true, and non is ignored if adapt=false.
- `measure`:  measurement function, See [`Vegas.montecarlo`](@ref), [`VegasMC.montecarlo`](@ref) and [`MCMC.montecarlo`](@ref) for more details.
- `measurefreq`: how often perform the measurement for ever `measurefreq` MC steps. If a measurement is expansive, you may want to make the measurement less frequent.
- `inplace`:  whether to use the inplace version of the integrand. Default is `false`, which is more convenient for integrand with a few return values but may cause type instability. Only useful for the :vegas and :vegasmc solver.
- `parallel`: :auto will automatically choose the best parallelization mode. :mpi will use MPI.jl to run the MC in parallel. :thread will use Threads.@threads to run the MC in parallel. Default is :auto.
- `kwargs`:   Keyword arguments. The supported keywords include,
  * `measure` and `measurefreq`: measurement function and how frequent it is called. 
  * If `config` is `nothing`, you may need to provide arguments for the `Configuration` constructor, check [`Configuration`](@ref) docs for more details.

# Examples
```julia-repl
integrate((x, c)->(x[1]^2+x[2]^2); var = Continuous(0.0, 1.0), dof = 2, print=-2, solver=:vegas)
Integral 1 = 0.6663652080622751 ± 0.000490978424216832   (chi2/dof = 0.645)

julia> integrate((x, f, c)-> (f[1] = x[1]^2+x[2]^2); var = Continuous(0.0, 1.0), dof = 2, print=-2, solver=:vegas, inplace=true)
Integral 1 = 0.6672083165915914 ± 0.0004919147870306026   (chi2/dof = 2.54)
```
"""
function integrate(integrand::Function;
    solver::Symbol=:vegasmc, # :mcmc, :vegas, or :vegasmc
    config::Union{Configuration,Nothing}=nothing,
    neval=1e4, # number of evaluations
    niter=10, # number of iterations
    block=16, # number of blocks
    print=-1, printio=stdout, save=0, saveio=nothing, timer=[],
    gamma=1.0, # learning rate of the reweight factor, only used in MCMC solver
    adapt=true, # whether to adapt the grid and the reweight factor
    debug=false, # whether to print debug information (type instability, etc.)
    reweight_goal::Union{Vector{Float64},Nothing}=nothing, # goal of visited steps of each integrand (include the normalization integral)
    ignore::Int=adapt ? 1 : 0, #ignore the first `ignore` iteractions in average
    measure::Union{Nothing,Function}=nothing,
    measurefreq::Int=1,
    inplace::Bool=false, # whether to use the inplace version of the integrand
    parallel::Symbol=:nothread, # :auto, :mpi, or :thread, or :serial
    kwargs...
)
    if isnothing(config)
        config = Configuration(; kwargs...)
    end

    if gamma > 1.0
        @warn(red("learning rate gamma should be less than 1.0"))
    end

    ############ initialized timer ####################################
    if print > 0
        push!(timer, StopWatch(print, report))
    end

    ########### initialized MPI #######################################
    (MPI.Initialized() == false) && MPI.Init()
    comm = MPI.COMM_WORLD

    ############# figure out the best parallelization mode ############
    parallel = MCUtility.choose_parallel(parallel)

    Nworker = MCUtility.nworker(parallel) # actual number of workers
    Nthread = MCUtility.nthreads(parallel) # numebr of threads, threads share the same memory

    ############# figure out evaluations in each block ################
    block = _standardize_block(block, Nworker)
    @assert block % Nworker == 0
    nevalperblock = neval ÷ block # number of evaluations per block

    ########## initialize the progress bar ############################
    #In the MPI/thread mode, progress will only need to track the progress of the root worker.
    Ntotal = niter * block ÷ Nworker
    progress = Progress(Ntotal; dt=(print >= 0 ? (0.5 + print) : 0.5), enabled=(print >= -1), showspeed=true, desc="Total iterations * blocks $(Ntotal): ", output=printio)

    # initialize temp variables
    configs = [deepcopy(config) for i in 1:Nthread] # configurations for each worker
    obsSum = [[zero(o) for o in config.observable] for _ in 1:Nthread] # sum of observables for each worker
    obsSquaredSum = [[zero(o) for o in config.observable] for _ in 1:Nthread] # sum of squared observables for each worker
    summedConfig = [deepcopy(config) for i in 1:Nthread] # summed configuration for each thread

    startTime = time()
    results=[]

    for iter in 1:niter

        for i in 1:Nthread
            fill!(obsSum[i], zero(obsSum[1][1]))
            fill!(obsSquaredSum[i], zero(obsSquaredSum[1][1]))
            clearStatistics!(summedConfig[i])
        end

        if parallel == :thread
            Threads.@threads for i = 1:block
                _block!(i ,configs, obsSum, obsSquaredSum, summedConfig, solver, progress,
                    integrand, nevalperblock, print, save, timer, debug,
                    measure, measurefreq, inplace, parallel)
            end
        else
            for i = 1:block
                _block!(i, configs, obsSum, obsSquaredSum, summedConfig, solver, progress,
                    integrand, nevalperblock, print, save, timer, debug,
                    measure, measurefreq, inplace, parallel)
            end
        end

        # println(configs[1].normalization)

        for i in 2:Nthread
            obsSum[1] += obsSum[i]
            obsSquaredSum[1] += obsSquaredSum[i]
            addConfig!(summedConfig[1], summedConfig[i])
        end

        #################### collect statistics  ####################################
        obsSum[1] = [MCUtility.MPIreduce(osum) for osum in obsSum[1]]
        obsSquaredSum[1] = [MCUtility.MPIreduce(osumsq) for osumsq in obsSquaredSum[1]]
        # collect all statistics to summedConfig of the root worker
        MPIreduceConfig!(summedConfig[1])


        if MCUtility.mpi_master() # only the master process will output results, no matter parallel = :mpi or :thread or :serial
            ################### self-learning ##########################################
            (solver == :mcmc || solver == :vegasmc) && doReweight!(summedConfig[1], gamma, reweight_goal)
        end

        ######################## syncronize between works ##############################

        # broadcast the reweight and var.histogram of the summedConfig of the root worker to two targets:
        # 1. config of the root worker
        # 2. config of the other workers
        # config.reweight = MPI.bcast(summedConfig[1].reweight, root, comm) # broadcast reweight factors to all workers
        # config.reweight = MCUtility.MPIbcast(summedConfig[1].reweight)
        # for (vi, var) in enumerate(config.var)
        #     _bcast_histogram!(var, summedConfig[1].var[vi], config, adapt)
        # end
        MPIbcastConfig!(summedConfig[1])

        for config in configs
            bcastConfig!(config, summedConfig[1])
            if adapt
                for v in config.var
                    Dist.train!(v)
                    Dist.initialize!(v, config)
                end
            end
        end

        if MCUtility.mpi_master() # only the master process will output results, no matter parallel = :mpi or :thread or :serial
            ##################### Extract Statistics  ################################
            mean, std = _mean_std(obsSum[1], obsSquaredSum[1], block)
            push!(results, (mean, std, configs[1])) # configs has tried grid
        end
        ################################################################################
    end

    ##########################  output results   ##############################
    if MCUtility.mpi_master() # only the master process will output results, no matter parallel = :mpi or :thread or :serial
        result = Result(results, ignore)
        if print >= 0
            report(result)
            (print > 0) && println(yellow("$(Dates.now()), Total time: $(time() - startTime) seconds."))
        end
        return result
    end
end

function _standardize_block(nblock, Nworker)
    #########  construct configurations for each block ################
    if nblock > Nworker
        nblock = (nblock ÷ Nworker) * Nworker # make Nblock % size ==0, error estimation assumes this relation
    else
        nblock = Nworker  # each worker should handle at least one block
    end
    return nblock
end

function _block!(iblock, configs, obsSum, obsSquaredSum, summedConfig, solver, progress, 
    integrand::Function, nevalperblock, print, save, timer, debug::Bool,
     measure::Union{Nothing, Function}, measurefreq, inplace, parallel)

    # rank core will run the block with the indexes: rank, rank+Nworker, rank+2Nworker, ...
    rank = MCUtility.rank(parallel)
    Nworker = MCUtility.nworker(parallel)
    # println(iblock, " ", rank, " ", Nworker)

    (iblock % Nworker != rank-1 ) && return

    config_n = configs[rank] # configuration for the worker with rank `rank`
    clearStatistics!(config_n) # reset statistics

    if solver == :vegasmc
        VegasMC.montecarlo(config_n, integrand, nevalperblock, print, save, timer, debug;
            measure=measure, measurefreq=measurefreq, inplace=inplace)
    elseif solver == :vegas
        Vegas.montecarlo(config_n, integrand, nevalperblock, print, save, timer, debug;
            measure=measure, measurefreq=measurefreq, inplace=inplace)
    elseif solver == :mcmc
        MCMC.montecarlo(config_n, integrand, nevalperblock, print, save, timer, debug;
            measure=measure, measurefreq=measurefreq)
    else
        error("Solver $solver is not supported!")
    end

    # println(config_n.normalization)


    if (config_n.normalization > 0.0) == false #in case config.normalization is not a number
        error("normalization of block $i is $(config_n.normalization), which is not positively defined!")
    end

    addConfig!(summedConfig[rank], config_n) # collect statistics from the config of each block to summedConfig

    for o in 1:config_n.N
        if obsSum[rank][o] isa AbstractArray
            m = config_n.observable[o] ./ config_n.normalization
            obsSum[rank][o] += m
            obsSquaredSum[rank][o] += (eltype(m) <: Complex) ? (@. (real(m))^2 + (imag(m))^2 * 1im) : m .^ 2
        else
            m = config_n.observable[o] / config_n.normalization
            obsSum[rank][o] += m
            obsSquaredSum[rank][o] += (eltype(m) <: Complex) ? (real(m))^2 + (imag(m))^2 * 1im : m^2
        end
    end

    if MCUtility.is_root(parallel)
        (print >= -1) && next!(progress)
    end
end

# function _bcast_histogram!(target::V, source::V, config, adapt) where {V}
#     comm = MPI.COMM_WORLD
#     root = 0 # rank of the root worker
#     if target isa Dist.CompositeVar
#         for (vi, v) in enumerate(target.vars)
#             _bcast_histogram!(v, source.vars[vi], config, adapt)
#         end
#     else
#         target.histogram = MPI.bcast(source.histogram, root, comm)
#         if adapt
#             Dist.train!(target)
#             Dist.initialize!(target, config)
#         end
#     end
# end

function _mean_std(obsSum, obsSquaredSum, block)
    function _sqrt(x)
        return x < 0.0 ? 0.0 : sqrt(x)
    end
    function elementwise(osquaredSum, mean, block)
        if block > 1
            if eltype(osquaredSum) <: Complex
                r_std = @. _sqrt((real.(osquaredSum) / block - real(mean)^2) / (block - 1))
                i_std = @. _sqrt((imag.(osquaredSum) / block - imag(mean)^2) / (block - 1))
                std = r_std + i_std * 1im
            else
                # println(obsSquaredSum, ", ", mean, ", ", block)
                std = @. _sqrt((osquaredSum / block - mean^2) / (block - 1))
            end
        else
            std = zero(osquaredSum) # avoid division by zero
        end
        return std
    end

    mean = [osum ./ block for osum in obsSum]
    std = [elementwise(obsSquaredSum[o], mean[o], block) for o in eachindex(obsSquaredSum)]
    return mean, std
end

function doReweight!(config, gamma, reweight_goal)
    avgstep = sum(config.visited)
    for (vi, v) in enumerate(config.visited)
        # if v > 1000
        if v <= 1
            config.reweight[vi] *= (avgstep)^gamma
        else
            config.reweight[vi] *= (avgstep / v)^gamma
        end
    end
    # println(config.visited)
    # println(config.reweight)
    if isnothing(reweight_goal) == false
        config.reweight .*= reweight_goal
    end
    # renoormalize all reweight to be (0.0, 1.0)
    config.reweight ./= sum(config.reweight)
    # avoid overreacting to atypically large reweighting factor
    # reweighting factor close to 1.0 will not be changed much
    # reweighting factor close to zero will be amplified significantly
    # Check Eq. (19) of https://arxiv.org/pdf/2009.05112.pdf for more detail
    # config.reweight = @. ((1 - config.reweight) / log(1 / config.reweight))^beta
    # config.reweight ./= sum(config.reweight)
end