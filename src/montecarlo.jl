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

- `measure`: function call to measure. It should accept an argument of the type `Configuration`, then manipulate observables `obs`. 

- `Nblock`: Number of blocks, each block corresponds to one Configuration struct. The tasks will automatically distributed to multi-process in MPI mode. If the numebr of workers N is larger than Nblock, then Nblock will be set to be N.

- `print`: -1 to not print anything, 0 to print minimal information, >0 to print summary for every `print` seconds

- `printio`: `io` to print the information

- `save`: -1 to not save anything, 0 to save observables `obs` in the end of sampling, >0 to save observables `obs` for every `save` seconds

- `saveio`: `io` to save

- `timer`: `StopWatch` other than print and save.

- `reweight = config.totalStep/10`: the MC steps before reweighting the integrands. Set to -1 if reweighting is not wanted.
"""
function sample(config::Configuration, integrand::Function, measure::Function=simple_measure;
    neval=1e4 * length(config.dof), # number of evaluations
    niter=10, # number of iterations
    block=16, # number of blocks
    alpha=0.5, # learning rate
    beta=1.0, # learning rate
    print=0, printio=stdout, save=0, saveio=nothing, timer=[])

    # println(reweight)

    if beta > 1.0
        @warn(red("beta should be less than 1.0"))
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
    # nevalperblock = neval ÷ block # number of evaluations per block
    nevalperblock = neval # number of evaluations per block

    results = []
    startTime = time()
    obsSum, obsSquaredSum = zero(config.observable), zero(config.observable)

    # configVec = Vector{Configuration}[]

    for iter in 1:niter

        obsSum *= 0
        obsSquaredSum *= 0

        for i = 1:block
            # MPI thread rank will run the block with the indexes: rank, rank+Nworker, rank+2Nworker, ...
            (i % Nworker != rank) && continue

            # reset!(config, config.reweight) # reset configuration, keep the previous reweight factors
            clearStatistics!(config) # reset configuration, keep the previous reweight factors

            config = montecarlo(config, integrand, measure, nevalperblock, print, save, timer)

            # summary = addStat(config, summary)  # collect MC information

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
        summedConfig = reduceConfig(config, root, comm)

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
            # return mean, std
            push!(results, (mean, std, summedConfig))
            # else # if not the root, return nothing
            # return nothing, nothing
            average(results)
            # println(average(results))

            ################### self-learning ##########################################
            doReweight!(summedConfig, beta)
            config.reweight = summedConfig.reweight
        end

        MPI.Bcast!(summedConfig.reweight, root, comm)
        # println(MPI.Comm_rank(comm), " reweight: ", summedConfig.reweight)
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

    updates = [changeIntegrand, swapVariable, changeVariable] # TODO: sample changeVariable more often
    for i = 2:length(config.var)
        push!(updates, changeVariable)
    end

    ########### MC simulation ##################################
    if (print >= 0)
        println(green("Seed $(config.seed) Start Simulation ..."))
    end
    startTime = time()

    for i = 1:neval
        config.neval += 1
        config.visited[config.curr] += 1
        _update = rand(config.rng, updates) # randomly select an update
        _update(config, integrand)
        if i % 10 == 0 && i >= neval / 100
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

    if (print >= 0)
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

function doReweight!(config, beta)
    avgstep = sum(config.visited)
    for (vi, v) in enumerate(config.visited)
        # if v > 1000
        if v <= 1
            config.reweight[vi] *= (avgstep)^beta
        else
            config.reweight[vi] *= (avgstep / v)^beta
        end
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
