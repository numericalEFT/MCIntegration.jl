"""
This algorithm combines Vegas with Markov-chain Monte Carlo.
For multiple integrands invoves multiple variables, it finds the best distribution
ansatz to fit them all together. In additional to the original integral, it also 
introduces a normalization integral with integrand ~ 1.

Assume f_0(x) and g_0(y) are the ansatz from the Vegas map for x and y, and we want
to calculate the integral f_1(x) and f_2(x, y)

Then the following distributions are sampled with Markov-chain Monte Carlo:
p_0(x, y) = f_0(x) * g_0(y) #normalization
p_1(x, y) = f_1(x) * g_0(y) #integrand 1 
p_2(x, y) = f_2(x, y)
NOTE: All three integral are measured at each Markov-chain Monte Carlo step!

The efficiency is significantly improved compared to the old DiagMC algorithm.
It is a few times slower than the Vegas algorithm at low dimensions.
However, the biggest problem is that the algorithm can fail if the integrand
exactly vanishes in some regime (e.g. circle area x^2+y^2<0).
"""

function montecarlo(config::Configuration{N,V,P,O,T}, integrand::Function, neval,
    print, save, timer;
    measurefreq=2, measure::Union{Nothing,Function}=nothing,
    kwargs...) where {N,V,P,O,T}

    if isnothing(measure)
        @assert (config.observable isa AbstractVector) && (length(config.observable) == config.N) && (eltype(config.observable) == T) "the default measure can only handle observable as Vector{$T} with $(config.N) elements!"
    end
    weights = zeros(T, N)
    relativeWeights = zeros(T, N)
    # padding probability for user and normalization integrands
    # should be something like [f_1(x)*g_0(y), f_2(x, y), 1*f_0(x)*g_0(y)], where f_0(x) and g_0(y) are the ansatz from the Vegas map
    # after padding, all integrands have the same dimension and have similiar probability distribution
    padding_probability = zeros(T, N + 1)
    ##############  initialization  ################################
    # don't forget to initialize the diagram weight
    initialize!(config, integrand)
    # for i in 1:10000
    #     initialize!(config, integrand)
    #     if (config.curr == config.norm) || abs(config.weights[config.curr]) > TINY
    #         break
    #     end
    # end
    # @assert (config.curr == config.norm) || abs(config.weights[config.curr]) > TINY "Cannot find the variables that makes the $(config.curr) integrand >1e-10"


    updates = [changeVariable,]
    # for i = 2:length(config.var)*2
    #     push!(updates, changeVariable)
    # end

    ########### MC simulation ##################################
    # if (print > 0)
    #     println(green("Seed $(config.seed) Start Simulation ..."))
    # end
    startTime = time()

    for i = 1:neval
        config.neval += 1
        _update = rand(config.rng, updates) # randomly select an update
        _update(config, integrand, weights, padding_probability)

        ######## accumulate variable and calculate variable probability #################
        for (vi, var) in enumerate(config.var)
            offset = var.offset
            for i in 1:N
                # need to make sure Δxᵢ*∫_xᵢ^xᵢ₊₁ dx f^2(x)dx is a constant
                # where  Δxᵢ ∝ 1/prop ∝ Jacobian for the vegas map
                # since the current weight is sampled with the probability density ∝ config.probability =\sum_i |f_i(x)|*reweight[i]
                # the estimator ∝ Δxᵢ*f^2(x)/(|f(x)|*reweight) = |f(x)|/prop/reweight
                f2 = abs(config.weights[i])^2 / Dist.probability(config, i)
                wf2 = f2 * Dist.padding_probability(config, i) / config.probability
                # push!(kwargs[:mesh], (f2, var[1]))
                for pos = 1:config.dof[i][vi]
                    # Dist.accumulate!(var, pos + offset, f2 / config.probability)

                    # the following accumulator has a similar performance
                    # this is because that with an optimial grid, |f(x)| ~ prop
                    # Dist.accumulate!(var, pos + offset, 1.0/ reweight
                    # Dist.accumulate!(var, pos + offset, 1.0 / config.reweight[config.curr])
                    Dist.accumulate!(var, pos + offset, wf2)
                    # Dist.accumulate!(var, pos + offset, abs(config.weights[i] * config.reweight[i]) / config.probability)
                end
            end
        end

        if i % measurefreq == 0 && i >= neval / 100
            ##############################################################################
            if isnothing(measure)
                for i in 1:N
                    # prob = Dist.delta_probability(config, config.curr; new=i)
                    # config.observable[i] += config.weights[i] * prob / config.probability
                    config.observable[i] += weights[i] * Dist.padding_probability(config, i) / config.probability
                    config.visited[i] += abs(weights[i] * Dist.padding_probability(config, i) * config.reweight[i]) / config.probability
                end
            else
                for i in 1:N
                    # prob = Dist.delta_probability(config, config.curr; new=i)
                    # config.relativeWeights[i] = config.weights[i] * prob / config.probability
                    relativeWeights[i] = weights[i] * Dist.padding_probability(config, i) / config.probability
                    config.visited[i] += abs(weights[i] * Dist.padding_probability(config, i) * config.reweight[i]) / config.probability
                end
                (fieldcount(V) == 1) ?
                measure(config.var[1], config.observable, relativeWeights, config) :
                measure(config.var, config.observable, relativeWeights, config)
            end
            # prob = Dist.delta_probability(config, config.curr; new=config.norm)
            # config.normalization += prob / config.probability
            config.normalization += 1.0 * Dist.padding_probability(config, config.norm) / config.probability
            config.visited[config.norm] += config.reweight[config.norm] * Dist.padding_probability(config, config.norm) / config.probability
            # push!(kwargs["mem"], (1.0 / config.probability, config.weights[i] / config.probability))
        end
        if i % 1000 == 0
            for t in timer
                check(t, config, neval)
            end
        end
    end

    return config
end

@inline function integrand_wrap(config, _integrand)
    return _integrand(config.var..., config)
end

function initialize!(config, integrand)
    for var in config.var
        Dist.initialize!(var, config)
    end

    # weights = integrand_wrap(config, integrand)
    weights = (length(config.var) == 1) ? integrand(config.var[1], config) : integrand(config.var, config)
    # config.probability = abs(weights[config.curr]) / Dist.probability(config, config.curr) * config.reweight[config.curr]
    # if config.curr == config.norm
    #     config.probability = config.reweight[config.curr]
    # else
    #     # config.probability = abs(weights[config.curr]) * config.reweight[config.curr] / Dist.probability(config, config.curr)
    #     config.probability = abs(weights[config.curr]) * config.reweight[config.curr]
    # end
    newProbability = config.reweight[config.norm] * Dist.padding_probability(config, config.norm) #normalization integral
    for i in 1:config.N #other integrals
        newProbability += abs(weights[i]) * config.reweight[i] * Dist.padding_probability(config, i)
    end
    config.probability = newProbability
    setWeight!(config, weights)
end

function setWeight!(config, weights)
    # weights can be a number of a vector of numbers
    # this function copy weights to a vector config.weights
    for i in eachindex(config.weights)
        config.weights[i] = weights[i]
    end
end
