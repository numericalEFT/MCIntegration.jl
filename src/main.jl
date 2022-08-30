"""
    function integrate(integrand::Function;
        config::Union{Configuration,Nothing}=nothing,
        measure::Function=simple_measure,
        neval=1e5, 
        niter=10, 
        block=16, 
        alpha=1.0, 
        print=-1, 
        printio=stdout,
        kwargs...
    )

 Calculate the integrals, collect statistics, and return a Result struct that contains the estimations and errors.

 # Remarks
 - User may run the MC in parallel using MPI. Simply run `mpiexec -n N julia userscript.jl` where `N` is the number of workers. In this mode, only the root process returns meaningful results. All other workers return `nothing, nothing`. User is responsible to handle the returning results properly. If you have multiple number of mpi version, you can use "mpiexecjl" in your "~/.julia/package/MPI/###/bin" to make sure the version is correct. See https://juliaparallel.github.io/MPI.jl/stable/configuration/ for more detail.
 - In the MC, a normalization diagram is introduced to normalize the MC estimates of the integrands. More information can be found in the link: https://kunyuan.github.io/QuantumStatistics.jl/dev/man/important_sampling/#Important-Sampling. User don't need to explicitly specify this normalization diagram.Internally, normalization diagram will be added to each table that is related to the integrands.

 # Arguments

- `integrand`: function call to evaluate the integrand. It should accept an argument of the type [`Configuration`](@ref), and return a weight. 
   Internally, MC only samples the absolute value of the weight. Therefore, it is also important to define Main.abs for the weight if its type is user-defined. 
- `config`: [`Configuration`](@ref) object to perform the MC integration. If `nothing`, it attempts to create a new one with Configuration(; kwargs...).
- `measure`: function call to measure. It should accept an argument the type [`Configuration`](@ref). Then you can accumulate the measurements with Configuration.obs. 
   If every integral is expected to be a float number, you can use MCIntegration.simple_measure as the default.
- `neval`: number of evaluations of the integrand per iteration. 
- `niter`: number of iterations. The reweight factor and the variables will be self-adapted after each iteration. 
- `block`: Number of blocks. Each block will be evaluated by about neval/block times. Each block is assumed to be statistically independent, and will be used to estimate the error. 
   In MPI mode, the blocks are distributed among the workers. If the numebr of workers N is larger than block, then block will be set to be N.
- `alpha`: Learning rate of the reweight factor after each iteraction. Note that alpha <=1, where alpha = 0 means no reweighting.  
- `print`: -1 to not print anything, 0 to print minimal information, >0 to print summary for every `print` seconds
- `printio`: `io` to print the information
- `kwargs`: keyword arguments. If `config` is `nothing`, you may need to provide arguments for the `Configuration` constructor, check [`Configuration`](@ref) docs for more details.

# Examples
```julia-repl
julia> integrate(c->(X=c.var[1]; X[1]^2+X[2]^2); var = (Continuous(0.0, 1.0), ), dof = [(2, ),], print=-1)
Integral 1 = 0.6830078240204353 ± 0.014960689298028415   (chi2/dof = 1.46)
```
"""
function integrate(integrand::Function;
    solver::Symbol=:MCMC, # :MCMC or :MC
    config::Union{Configuration,Nothing}=nothing,
    neval=1e4, # number of evaluations
    niter=10, # number of iterations
    block=16, # number of blocks
    alpha=1.0, # learning rate of the reweight factor
    print=0, printio=stdout, save=0, saveio=nothing, timer=[],
    kwargs...
)
    if isnothing(config)
        config = Configuration(; kwargs...)
    end

    if alpha > 1.0
        @warn(red("learning rate alpha should be less than 1.0"))
    end

    ############ initialized timer ####################################
    if print > 0
        push!(timer, StopWatch(print, summary))
    end

    ########### initialized MPI #######################################
    (MPI.Initialized() == false) && MPI.Init()
    comm = MPI.COMM_WORLD
    Nworker = MPI.Comm_size(comm)  # number of MPI workers
    rank = MPI.Comm_rank(comm)  # rank of current MPI worker
    root = 0 # rank of the root worker
    # MPI.Barrier(comm)

    #########  construct configurations for each block ################
    if block > Nworker
        block = (block ÷ Nworker) * Nworker # make Nblock % size ==0, error estimation assumes this relation
    else
        block = Nworker  # each worker should handle at least one block
    end
    @assert block % Nworker == 0
    nevalperblock = neval ÷ block # number of evaluations per block
    # nevalperblock = neval # number of evaluations per block

    results = []
    obsSum, obsSquaredSum = zero(config.observable), zero(config.observable)

    # configVec = Vector{Configuration}[]

    #In the MPI mode, progress will only need to track the progress of the root worker.
    Ntotal = niter * block ÷ Nworker
    progress = Progress(Ntotal; dt=(0.5 + print), enabled=(print >= 0), showspeed=true, desc="Total iterations * blocks $(Ntotal): ", output=printio)

    startTime = time()
    for iter in 1:niter

        obsSum *= 0
        obsSquaredSum *= 0
        # summed configuration of all blocks, but changes in each iteration
        summedConfig = deepcopy(config)

        for i = 1:block
            # MPI thread rank will run the block with the indexes: rank, rank+Nworker, rank+2Nworker, ...
            (i % Nworker != rank) && continue

            # reset!(config, config.reweight) # reset configuration, keep the previous reweight factors
            clearStatistics!(config) # reset statistics

            if solver == :MCMC
                config = MCMC.montecarlo(config, integrand, nevalperblock, print, save, timer; kwargs...)
            elseif solver == :MC
            else
                error("Solver $solver is not supported!")
            end

            addConfig!(summedConfig, config) # collect statistics from the config of each block to summedConfig

            if (config.normalization > 0.0) == false #in case config.normalization is not a number
                error("normalization of block $i is $(config.normalization), which is not positively defined!")
            end

            if typeof(obsSum) <: AbstractArray
                obsSum .+= config.observable ./ config.normalization
                if eltype(obsSquaredSum) <: Complex  #ComplexF16, ComplexF32 or ComplexF64 array
                    obsSquaredSum .+= (real.(config.observable) ./ config.normalization) .^ 2
                    obsSquaredSum .+= (imag.(config.observable) ./ config.normalization) .^ 2 * 1im
                else
                    obsSquaredSum .+= (config.observable ./ config.normalization) .^ 2
                end
            else
                obsSum += config.observable / config.normalization
                if typeof(obsSquaredSum) <: Complex
                    obsSquaredSum += (real(config.observable) / config.normalization)^2
                    obsSquaredSum += (imag(config.observable) / config.normalization)^2 * 1im
                else
                    obsSquaredSum += (config.observable / config.normalization)^2
                end
            end

            if MPI.Comm_rank(comm) == root
                (print >= 0) && next!(progress)
            end
        end
        #################### collect statistics  ####################################
        MCUtility.MPIreduce(obsSum)
        MCUtility.MPIreduce(obsSquaredSum)
        # collect all statistics to summedConfig of the root worker
        MPIreduceConfig!(summedConfig, root, comm)

        if MPI.Comm_rank(comm) == root
            ##################### Extract Statistics  ################################
            mean = obsSum ./ block
            if eltype(obsSquaredSum) <: Complex
                r_std = @. sqrt((real.(obsSquaredSum) / block - real(mean)^2) / (block - 1))
                i_std = @. sqrt((imag.(obsSquaredSum) / block - imag(mean)^2) / (block - 1))
                std = r_std + i_std * 1im
            else
                std = @. sqrt((obsSquaredSum / block - mean^2) / (block - 1))
            end
            push!(results, (mean, std, summedConfig))

            ################### self-learning ##########################################
            (solver == :MCMC) && MCMC.doReweight!(summedConfig, alpha)
            for var in summedConfig.var
                Dist.train!(var)
            end
        end

        ######################## syncronize between works ##############################

        # broadcast the reweight and var.histogram of the summedConfig of the root worker to two targets:
        # 1. config of the root worker
        # 2. config of the other workers
        config.reweight = MPI.bcast(summedConfig.reweight, root, comm) # broadcast reweight factors to all workers
        for vi in 1:length(config.var)
            config.var[vi].histogram = MPI.bcast(summedConfig.var[vi].histogram, root, comm)
            Dist.train!(config.var[vi])
        end
        ################################################################################
        if MPI.Comm_rank(comm) == root
            if print >= 0
                # println(green("Iteration $iter is done. $(time() - startTime) seconds passed."))
                # next!(progress)
            end
        end
    end
    ################################ IO ######################################
    if MPI.Comm_rank(comm) == root
        result = Result(results)
        if print >= 0
            report(result)
            (print > 0) && println(yellow("$(Dates.now()), Total time: $(time() - startTime) seconds."))
        end
        return result
    end
end