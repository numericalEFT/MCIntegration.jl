"""
Monte Carlo Calculator for Diagrams
"""

using Random, MPI
using LinearAlgebra
using StaticArrays, Printf, Dates
using Graphs
using ProgressMeter
using .MCUtility
const RNG = Random.GLOBAL_RNG

include("configuration.jl")
include("variable.jl")
include("sampler.jl")
include("updates.jl")
include("statistics.jl")

"""

    function integrate(integrand::Function;
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
- `measure`: function call to measure. It should accept an argument the type [`Configuration`](@ref). Then you can accumulate the measurements with Configuration.obs. 
   If every integral is expected to be a float number, you can use MCIntegration.simple_measure as the default.
- `neval`: number of evaluations of the integrand per iteration. 
- `niter`: number of iterations. The reweight factor and the variables will be self-adapted after each iteration. 
- `block`: Number of blocks. Each block will be evaluated by about neval/block times. Each block is assumed to be statistically independent, and will be used to estimate the error. 
   In MPI mode, the blocks are distributed among the workers. If the numebr of workers N is larger than block, then block will be set to be N.
- `alpha`: Learning rate of the reweight factor after each iteraction. Note that alpha <=1, where alpha = 0 means no reweighting.  
- `print`: -1 to not print anything, 0 to print minimal information, >0 to print summary for every `print` seconds
- `printio`: `io` to print the information
- `kwargs`: keyword arguments. 
   If `config` is specified, then `config` will be used as the [`Configuration`](@ref). Otherwise, a new [`Configuration`](@ref) object will be created with the constructor Configuration(; kwargs...).
   In the latter case, you may need to specifiy additional arguments for the `Configuration` constructor, check [`Configuration`](@ref) docs for more details.

# Examples
```julia-repl
julia> integrate(c->(X=c.var[1]; X[1]^2+X[2]^2); var = (Continuous(0.0, 1.0), ), dof = [(2, ),], print=-1)
==================================     Integral-1    ==============================================
  iter          integral                            wgt average                          chi2/dof
---------------------------------------------------------------------------------------------------
     1       0.75635674 ± 0.033727463             0.75635674 ± 0.033727463                 0.0000
     2       0.63865386 ± 0.037056202             0.70302829 ± 0.024942894                 5.5180
     3       0.76433252 ± 0.06486697              0.71092504 ± 0.023281055                 3.1480
     4       0.64730886 ± 0.044154018             0.69708628 ± 0.020593731                 2.6401
     5       0.70840759 ± 0.043918356             0.69912689 ± 0.018645635                 1.9937
     6       0.66853011 ± 0.039370893             0.69352162 ± 0.016851385                 1.6936
     7       0.75108782 ± 0.046025832             0.70032623 ± 0.015824115                 1.6413
     8       0.67506441 ± 0.053890967             0.69832105 ± 0.015183104                 1.4357
     9       0.71884473 ± 0.046650952              0.7002868 ± 0.014437688                 1.2781
    10       0.69904531 ± 0.033491092             0.70009224 ± 0.013258207                 1.1362
---------------------------------------------------------------------------------------------------
Integral-1 = 0.700092241210273 ± 0.013258207327344304
```
"""
function integrate(integrand::Function;
    measure::Function=simple_measure,
    neval=1e4, # number of evaluations
    niter=10, # number of iterations
    block=16, # number of blocks
    alpha=1.0, # learning rate of the reweight factor
    print=0, printio=stdout, save=0, saveio=nothing, timer=[],
    kwargs...
)
    if haskey(kwargs, "config")
        config = pop!(kwargs, "config")
    else
        config = Configuration(; kwargs...)
    end
    return sample(config, integrand, measure;
        neval=neval,
        niter=niter,
        block=block,
        alpha=alpha,
        print=print, printio=printio, save=save, saveio=saveio, timer=timer, kwargs...)
end

function sample(config::Configuration, integrand::Function, measure::Function=simple_measure;
    neval=1e4 * length(config.dof), # number of evaluations
    niter=10, # number of iterations
    block=16, # number of blocks
    alpha=1.0, # learning rate of the reweight factor
    print=-1, printio=stdout, save=0, saveio=nothing, timer=[],
    kwargs...
)

    # println(reweight)

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
    # if print > 0
    Ntotal = niter * block ÷ Nworker
    progress = Progress(Ntotal; dt=(0.5 + print), enabled=(print >= 0), showspeed=true, desc="Total iterations * blocks $(Ntotal): ", output=printio)
    # else
    #     progress = Progress(niter; dt=(0.1 + print), enabled=(print >= 0), showspeed=true, desc="Total iterations $(niter): ", output=printio)
    # end

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

            config = montecarlo(config, integrand, measure, nevalperblock, print, save, timer)

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
                if print >= 0
                    next!(progress)
                    # println()
                end
            end
        end
        #################### collect statistics  ####################################
        MPIreduce(obsSum)
        MPIreduce(obsSquaredSum)
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
            # MPI.Finalize()
            push!(results, (mean, std, summedConfig))
            # average(results)

            ################### self-learning ##########################################
            doReweight!(summedConfig, alpha)
            for var in summedConfig.var
                train!(var)
            end
        end

        ######################## syncronize between works ##############################
        # println(MPI.Comm_rank(comm), " reweight: ", config.reweight)

        # broadcast the reweight and var.histogram of the summedConfig of the root worker to two targets:
        # 1. config of the root worker
        # 2. config of the other workers
        config.reweight = MPI.bcast(summedConfig.reweight, root, comm) # broadcast reweight factors to all workers
        for vi in 1:length(config.var)
            config.var[vi].histogram = MPI.bcast(summedConfig.var[vi].histogram, root, comm)
            train!(config.var[vi])
        end
        # if MPI.Comm_rank(comm) == 1
        #     println("1 reweight: ", config.reweight, " vs ", summedConfig.reweight)
        #     println("1 var: ", config.var[1].histogram, " vs ", summedConfig.var[1].histogram)
        # end
        # if MPI.Comm_rank(comm) == 0
        #     sleep(1)
        #     println("0 reweight: ", config.reweight, " vs ", summedConfig.reweight)
        #     println("0 var: ", config.var[1].histogram, " vs ", summedConfig.var[1].histogram)
        # end
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
        # if (print >= 0)
        # summary(results[end][3], neval)
        # end
        if print >= 0
            summary(result)
            if print > 0
                println(yellow("$(Dates.now()), Total time: $(time() - startTime) seconds."))
                # println(green("Cost $(time() - startTime) seconds."))
            end
        end
        return result
        # return result.mean, result.stdev
    end
end

function montecarlo(config::Configuration, integrand::Function, measure::Function, neval, print, save, timer)
    ##############  initialization  ################################
    # don't forget to initialize the diagram weight
    config.absWeight = abs(integrand(config))


    # updates = [changeIntegrand,] # TODO: sample changeVariable more often
    updates = [changeIntegrand, swapVariable, changeVariable] # TODO: sample changeVariable more often
    for i = 2:length(config.var)*2
        push!(updates, changeVariable)
    end

    ########### MC simulation ##################################
    # if (print > 0)
    #     println(green("Seed $(config.seed) Start Simulation ..."))
    # end
    startTime = time()

    for i = 1:neval
        config.neval += 1
        config.visited[config.curr] += 1
        _update = rand(config.rng, updates) # randomly select an update
        _update(config, integrand)
        if i % 10 == 0 && i >= neval / 100

            ######## accumulate variable #################
            if config.curr != config.norm
                for (vi, var) in enumerate(config.var)
                    offset = var.offset
                    for pos = 1:config.dof[config.curr][vi]
                        accumulate!(var, pos + offset)
                    end
                end
            end
            ###############################################

            if config.curr == config.norm # the last diagram is for normalization
                config.normalization += 1.0 / config.reweight[config.norm]
            else
                if measure == simple_measure
                    simple_measure(config, integrand)
                else
                    measure(config)
                end
            end
        end
        if i % 1000 == 0
            for t in timer
                check(t, config, neval)
            end
        end
    end

    # if (print > 0)
    #     println(green("Seed $(config.seed) End Simulation. Cost $(time() - startTime) seconds."))
    # end

    return config
end

function simple_measure(config, integrand)
    # factor = 1.0 / config.reweight[config.curr]
    # weight = integrand(config)
    if config.observable isa AbstractVector
        # config.observable[config.curr] += weight / abs(weight) * factor
        config.observable[config.curr] += config.relativeWeight
    elseif config.observable isa AbstractFloat
        # config.observable += weight / abs(weight) * factor
        config.observable += config.relativeWeight
    else
        error("simple_measure can only be used with AbstractVector or AbstractFloat observables")
    end
end

function doReweight!(config, alpha)
    avgstep = sum(config.visited)
    for (vi, v) in enumerate(config.visited)
        # if v > 1000
        if v <= 1
            config.reweight[vi] *= (avgstep)^alpha
        else
            config.reweight[vi] *= (avgstep / v)^alpha
        end
    end
    config.reweight .*= config.reweight_goal
    # renoormalize all reweight to be (0.0, 1.0)
    config.reweight ./= sum(config.reweight)
    # avoid overreacting to atypically large reweighting factor
    # reweighting factor close to 1.0 will not be changed much
    # reweighting factor close to zero will be amplified significantly
    # Check Eq. (19) of https://arxiv.org/pdf/2009.05112.pdf for more detail
    # config.reweight = @. ((1 - config.reweight) / log(1 / config.reweight))^beta
    # config.reweight ./= sum(config.reweight)
end

# function doReweight!(config, alpha)
#     avgstep = sum(config.visited) / length(config.visited)
#     for (vi, v) in enumerate(config.visited)
#         if v > 1000
#             config.reweight[vi] *= avgstep / v
#             if config.reweight[vi] < 1e-10
#                 config.reweight[vi] = 1e-10
#             end
#         end
#     end
#     # renoormalize all reweight to be (0.0, 1.0)
#     config.reweight .= config.reweight ./ sum(config.reweight)
#     # dample reweight factor to avoid rapid, destabilizing changes
#     # reweight factor close to 1.0 will not be changed much
#     # reweight factor close to zero will be amplified significantly
#     # Check Eq. (19) of https://arxiv.org/pdf/2009.05112.pdf for more detail
#     config.reweight = @. ((1 - config.reweight) / log(1 / config.reweight))^2.0
# end
