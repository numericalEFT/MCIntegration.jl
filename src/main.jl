"""
    function integrate(integrand::Function;
        solver::Symbol=:vegas, # :mcmc, :vegas, or :vegasmc
        config::Union{Configuration,Nothing}=nothing,
        neval=1e4, 
        niter=10, 
        block=16, 
        verbose=-1, 
        gamma=1.0, 
        adapt=true,
        debug=false, 
        reweight_goal::Union{Vector{Float64},Nothing}=nothing, 
        ignore::Int=adapt ? 1 : 0,
        measure::Union{Nothing,Function}=nothing,
        measurefreq::Int=1,
        inplace::Bool=false,
        parallel::Symbol=:nothread,
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
- `verbose`:  < -1 to not print anything; -1 to print minimal information; 0 to print the iteration history in the end; >0 to print MC configuration for every `print` seconds and print the iteration history in the end.
- `gamma`:    Learning rate of the reweight factor after each iteraction. Note that gamma <=1, where gamma = 0 means no reweighting.  
- `adapt`:    Whether to adapt the grid and the reweight factor.
- `debug`:    Whether to print debug information (type instability, float overflow etc.)
- `reweight_goal`: The expected distribution of visited times for each integrand after reweighting . If not set, then all factors will be initialized with one. Only useful for the :mcmc solver. 
- `ignore`:   ignore the iteration until the `ignore` round. By default, the first iteration is igonred if adapt=true, and non is ignored if adapt=false.
- `measure`:  measurement function, See [`Vegas.montecarlo`](@ref), [`VegasMC.montecarlo`](@ref) and [`MCMC.montecarlo`](@ref) for more details.
- `measurefreq`: how often perform the measurement for ever `measurefreq` MC steps. If a measurement is expansive, you may want to make the measurement less frequent.
- `inplace`:  whether to use the inplace version of the integrand. Default is `false`, which is more convenient for integrand with a few return values but may cause type instability. Only useful for the :vegas and :vegasmc solver.
- `parallel`: :thread will use Threads.@threads to run different blocks in parallel. Default is :nothread.
- `kwargs`:   Keyword arguments. The supported keywords include,
  * `measure` and `measurefreq`: measurement function and how frequent it is called. 
  * If `config` is `nothing`, you may need to provide arguments for the `Configuration` constructor, check [`Configuration`](@ref) docs for more details.

# Examples
```julia-repl
integrate((x, c)->(x[1]^2+x[2]^2); var = Continuous(0.0, 1.0), dof = 2, verbose=-2, solver=:vegas)
Integral 1 = 0.6663652080622751 ± 0.000490978424216832   (chi2/dof = 0.645)

julia> integrate((x, f, c)-> (f[1] = x[1]^2+x[2]^2); var = Continuous(0.0, 1.0), dof = 2, verbose=-2, solver=:vegas, inplace=true)
Integral 1 = 0.6672083165915914 ± 0.0004919147870306026   (chi2/dof = 2.54)
```
"""
function integrate(integrand::Function;
    solver::Symbol=:vegasmc, # :mcmc, :vegas, or :vegasmc
    config::Union{Configuration,Nothing}=nothing,
    neval=1e4, # number of evaluations
    niter=10, # number of iterations
    block=16, # number of blocks
    verbose=-1, # verbose level
    gamma=1.0, # learning rate of the reweight factor, only used in MCMC solver
    adapt=true, # whether to adapt the grid and the reweight factor
    debug=false, # whether to print debug information (type instability, etc.)
    reweight_goal::Union{Vector{Float64},Nothing}=nothing, # goal of visited steps of each integrand (include the normalization integral)
    ignore::Int=adapt ? 1 : 0, #ignore the first `ignore` iteractions in average
    measure::Union{Nothing,Function}=nothing,
    measurefreq::Int=1,
    inplace::Bool=false, # whether to use the inplace version of the integrand
    parallel::Symbol=:nothread, # :thread or :nothread
    print=-1, printio=stdout, timer=[],
    kwargs...
)

    # we use print instead of verbose for historical reason
    print = maximum([print, verbose])

    if isnothing(config)
        config = Configuration(; kwargs...)
    end

    for i in eachindex(config.maxdof)
        @assert config.maxdof[i] + 2 <= poolsize(config.var[i]) "maxdof should be less than the length of var"
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

    # numebr of threads within MCIntegration, threads share the same memory
    # >1 only in the :thread parallel mode
    Nthread = MCUtility.nthreads(parallel)

    ############# figure out evaluations in each block ################
    nevalperblock, block = _standardize_block(neval, block)
    @assert block % MCUtility.mpi_nprocs() == 0

    ########## initialize the progress bar ############################
    #In the MPI/thread mode, progress will only need to track the progress of the root worker.
    Ntotal = niter * block ÷ MCUtility.nworker(parallel)
    progress = Progress(Ntotal; dt=(print >= 0 ? (0.5 + print) : 0.5), enabled=(print >= -1), showspeed=true, desc="Total iterations * blocks $(Ntotal): ", output=printio)

    # initialize temp variables
    configs = [deepcopy(config) for _ in 1:Nthread] # configurations for each thread
    summedConfig = [deepcopy(config) for _ in 1:Nthread] # summed configuration for each thread from different blocks
    obsSum = [[zero(o) for o in config.observable] for _ in 1:Nthread] # sum of observables for each worker
    obsSquaredSum = [[zero(o) for o in config.observable] for _ in 1:Nthread] # sum of squared observables for each worker

    for _config in configs
        reset_seed!(_config, rand(config.rng, 1:1000000)) # reset the seed for each thread
    end

    startTime = time()
    results = []

    for iter in 1:niter

        for i in 1:Nthread
            for j in eachindex(obsSum[i])
                obsSum[i][j] = zero(obsSum[i][j])
                obsSquaredSum[i][j] = zero(obsSquaredSum[i][j])
            end
            clearStatistics!(summedConfig[i])
        end

        for _ in MCUtility.mpi_nprocs()
            if parallel == :thread
                Threads.@threads for _ in 1:block/MCUtility.mpi_nprocs()
                    _block!(configs, obsSum, obsSquaredSum, summedConfig, solver, progress,
                        integrand, nevalperblock, print, timer, debug,
                        measure, measurefreq, inplace, parallel)
                end
            else
                for _ in 1:block/MCUtility.mpi_nprocs()
                    _block!(configs, obsSum, obsSquaredSum, summedConfig, solver, progress,
                        integrand, nevalperblock, print, timer, debug,
                        measure, measurefreq, inplace, parallel)
                end
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

        ######################## self-learning #########################################
        (solver == :mcmc || solver == :vegasmc) && doReweightMPI!(summedConfig[1], gamma, reweight_goal, comm)

        ######################## syncronize between works ##############################

        # broadcast the reweight and var.histogram of the summedConfig[1] from the root to all workers
        MPIbcastConfig!(summedConfig[1])

        for config in configs
            # broadcast the reweight and var.histogram of the summedConfig[1] to config of all threads
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

function _standardize_block(neval, nblock)
    #########  construct configurations for each block ################
    @assert neval > nblock "neval=$neval should be larger than nblock = $nblock"
    # standardize nblock according to MPI workers
    Nworker = MCUtility.mpi_nprocs() # number of MPI workers
    if nblock > Nworker
        # make Nblock % nworker ==0, error estimation assumes this relation
        nblock = (nblock ÷ Nworker) * Nworker
    else
        nblock = Nworker  # each worker should handle at least one block
    end

    nevalperblock = neval ÷ nblock # make sure each block has the same nevalperblock, error estimation assumes this relation
    return nevalperblock, nblock
end

function _block!(configs, obsSum, obsSquaredSum, summedConfig,
    solver, progress,
    integrand::Function, nevalperblock, print, timer, debug::Bool,
    measure::Union{Nothing,Function}, measurefreq, inplace, parallel)

    rank = MCUtility.threadid(parallel)
    # println(rank)

    config_n = configs[rank] # configuration for the worker with thread id `rank`
    clearStatistics!(config_n) # reset statistics

    if solver == :vegasmc
        VegasMC.montecarlo(config_n, integrand, nevalperblock, print, timer, debug;
            measure=measure, measurefreq=measurefreq, inplace=inplace)
    elseif solver == :vegas
        Vegas.montecarlo(config_n, integrand, nevalperblock, print, timer, debug;
            measure=measure, measurefreq=measurefreq, inplace=inplace)
    elseif solver == :mcmc
        MCMC.montecarlo(config_n, integrand, nevalperblock, print, timer, debug;
            measure=measure, measurefreq=measurefreq)
    else
        error("Solver $solver is not supported!")
    end

    # println(config_n.normalization)


    if (config_n.normalization > 0.0) == false #in case config.normalization is not a number
        error("Block normalization = $(config_n.normalization) is not positively defined!")
    end

    addConfig!(summedConfig[rank], config_n) # collect statistics from the config of each block to summedConfig

    for o in 1:config_n.N
        if obsSum[rank][o] isa AbstractArray
            m = config_n.observable[o] ./ config_n.normalization
            obsSum[rank][o] += m
            obsSquaredSum[rank][o] += (eltype(m) <: Complex) ? (@. real(m) * real(m) + imag(m) * imag(m) * 1im) : m .* m
            # avoid ^2 operator because it may not be defined for user defined types
        else
            m = config_n.observable[o] / config_n.normalization
            obsSum[rank][o] += m
            obsSquaredSum[rank][o] += (eltype(m) <: Complex) ? real(m) * real(m) + imag(m) * imag(m) * 1im : m^2
            # avoid ^2 operator because it may not be defined for user defined types
        end
    end

    if MCUtility.is_root(parallel)
        (print >= -1) && next!(progress)
    end
end

#obsSum or obsSquaredSum can be scalar or vector of float or complex
#the return value is always a vector of float or complex
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

    # println(obsSum)
    mean = [osum / block for osum in obsSum]
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
    if !isnothing(reweight_goal) # Apply reweight_goal if provided
        # config.reweight .*= reweight_goal
        config.reweight .*= reweight_goal ./ sum(reweight_goal)
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

function doReweightMPI!(config::Configuration, gamma, reweight_goal::Union{Vector{Float64},Nothing}, comm::MPI.Comm)
    if MCUtility.mpi_master()
        # only the master process will output results, no matter parallel = :mpi or :thread or :serial
        doReweight!(config, gamma, reweight_goal)
    end
    reweight_array = Vector{Float64}(config.reweight)
    MPI.Bcast!(reweight_array, 0, comm)
    config.reweight .= reweight_array
end