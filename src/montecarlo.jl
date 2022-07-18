"""
Monte Carlo Calculator for Diagrams
"""

using Random, MPI
using LinearAlgebra
using StaticArrays, Printf, Dates
using Graphs
using .MCUtility
const RNG = Random.GLOBAL_RNG

include("configuration.jl")
include("variable.jl")
include("sampler.jl")
include("updates.jl")
include("statistics.jl")

"""

    sample(config::Configuration, integrand::Function, measure::Function; Nblock=16, print=0, printio=stdout, save=0, saveio=nothing, timer=[])

 sample the integrands, collect statistics, and return the expected values and errors.

 # Remarks
 - User may run the MC in parallel using MPI. Simply run `mpiexec -n N julia userscript.jl` where `N` is the number of workers. In this mode, only the root process returns meaningful results. All other workers return `nothing, nothing`. User is responsible to handle the returning results properly. If you have multiple number of mpi version, you can use "mpiexecjl" in your "~/.julia/package/MPI/###/bin" to make sure the version is correct. See https://juliaparallel.github.io/MPI.jl/stable/configuration/ for more detail.

 - In the MC, a normalization diagram is introduced to normalize the MC estimates of the integrands. More information can be found in the link: https://kunyuan.github.io/QuantumStatistics.jl/dev/man/important_sampling/#Important-Sampling. User don't need to explicitly specify this normalization diagram.Internally, normalization diagram will be added to each table that is related to the integrands.

 # Arguments

 - `config`: Configuration struct

 - `integrand`: function call to evaluate the integrand. It should accept an argument of the type `Configuration`, and return a weight. 
    Internally, MC only samples the absolute value of the weight. Therefore, it is also important to define Main.abs for the weight if its type is user-defined. 

- `measure`: function call to measure. It should accept an argument of the type `Configuration`, then manipulate observables `obs`. By default, the function MCIntegration.simple_measure will be used.

- `neval`: number of evaluations of the integrand per iteration. By default, it is set to 1e4 * length(config.dof).

- `niter`: number of iterations. The reweight factor and the variables will be self-adapted after each iteration. By default, it is set to 10.

- `block`: Number of blocks. Each block will be evaluated by about neval/block times. The results from the blocks will be assumed to be statistically independent, and will be used to estimate the error.
   The tasks will automatically distributed to multi-process in MPI mode. If the numebr of workers N is larger than block, then block will be set to be N.
   By default, it is set to 16.

- `alpha`: Learning rate of the reweight factor after each iteraction. Note that alpha <=1, where alpha = 0 means no reweighting.  

- `print`: -1 to not print anything, 0 to print minimal information, >0 to print summary for every `print` seconds

- `printio`: `io` to print the information

- `save`: -1 to not save anything, 0 to save observables `obs` in the end of sampling, >0 to save observables `obs` for every `save` seconds

- `saveio`: `io` to save

- `timer`: `StopWatch` other than print and save.
"""
function sample(config::Configuration, integrand::Function, measure::Function=simple_measure;
    neval=1e4 * length(config.dof), # number of evaluations
    niter=10, # number of iterations
    block=16, # number of blocks
    alpha=1.0, # learning rate of the reweight factor
    print=0, printio=stdout, save=0, saveio=nothing, timer=[])

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
    startTime = time()
    obsSum, obsSquaredSum = zero(config.observable), zero(config.observable)

    # configVec = Vector{Configuration}[]

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
        if print >= 0
            println(green("Iteration $iter is done. $(time() - startTime) seconds passed."))
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
            println(red("Cost $(time() - startTime) seconds."))
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
    if (print > 0)
        println(green("Seed $(config.seed) Start Simulation ..."))
    end
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

    if (print > 0)
        println(green("Seed $(config.seed) End Simulation. Cost $(time() - startTime) seconds."))
    end

    return config
end

function simple_measure(config, integrand)
    factor = 1.0 / config.reweight[config.curr]
    weight = integrand(config)
    if config.observable isa AbstractVector
        config.observable[config.curr] += weight / abs(weight) * factor
    elseif config.observable isa AbstractFloat
        config.observable += weight / abs(weight) * factor
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
