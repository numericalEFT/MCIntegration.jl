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
    print=0, save=0, timer=[], debug=false;
    measurefreq=2, measure::Union{Nothing,Function}=nothing,
    kwargs...) where {N,V,P,O,T}

    ################## test integrand type stability ######################
    if debug
        if (length(config.var) == 1)
            MCUtility.test_type_stability(integrand, (config.var[1], config))
        else
            MCUtility.test_type_stability(integrand, (config.var, config))
        end
    end
    #######################################################################

    if isnothing(measure)
        @assert (config.observable isa AbstractVector) && (length(config.observable) == config.N) && (eltype(config.observable) == T) "the default measure can only handle observable as Vector{$T} with $(config.N) elements!"
    end
    weights = zeros(T, N)
    relativeWeights = zeros(T, N)
    # padding probability for user and normalization integrands
    # should be something like [f_1(x)*g_0(y), f_2(x, y), 1*f_0(x)*g_0(y)], where f_0(x) and g_0(y) are the ansatz from the Vegas map
    # after padding, all integrands have the same dimension and have similiar probability distribution
    padding_probability = zeros(Float64, N + 1)
    padding_probability_cache = zeros(Float64, N + 1) #used for cache the padding probability of the proposal configuration
    ##############  initialization  ################################
    # don't forget to initialize the diagram weight
    for var in config.var
        Dist.initialize!(var, config)
    end

    _weights = (length(config.var) == 1) ? integrand(config.var[1], config) : integrand(config.var, config)

    padding_probability .= [Dist.padding_probability(config, i) for i in 1:N+1]
    probability = config.reweight[config.norm] * padding_probability[config.norm] #normalization integral
    for i in 1:config.N #other integrals
        weights[i] = _weights[i]
        probability += abs(_weights[i]) * config.reweight[i] * padding_probability[i]
    end
    # config.probability = newProbability


    updates = [changeVariable,]
    # for i = 2:length(config.var)*2
    #     push!(updates, changeVariable) #add other updates
    # end
    if debug
        for _update in updates
            MCUtility.test_type_stability(_update, (config, integrand, probability,
                weights, padding_probability, padding_probability_cache))
        end
    end

    ########### MC simulation ##################################
    startTime = time()

    for ne = 1:neval
        config.neval += 1
        _update = rand(config.rng, updates) # randomly select an update
        probability = _update(config, integrand, probability,
            weights, padding_probability, padding_probability_cache)
        if debug && (isfinite(probability) == false)
            @warn("integrand probability = $(probability) is not finite at step $(neval)")
        end
        # WARNING: Don't turn it on, because some integral may actually vanish (for example, circle are) 
        # if debug && (all(x -> isfinite(x), weights)) 
        #     @warn("integrand = $(weights) is not all finite at step $(neval)")
        # end

        ######## accumulate variable and calculate variable probability #################
        for i in 1:N
            # need to make sure Δxᵢ*∫_xᵢ^xᵢ₊₁ dx f^2(x)dx is a constant
            # where  Δxᵢ ∝ 1/prop ∝ Jacobian for the vegas map
            # since the current weight is sampled with the probability density ∝ config.probability =\sum_i |f_i(x)|*reweight[i]
            # the estimator ∝ Δxᵢ*f^2(x)/(|f(x)|*reweight) = |f(x)|/prop/reweight
            f2 = abs(weights[i])^2 / Dist.probability(config, i)
            wf2 = f2 * padding_probability[i] / probability
            for (vi, var) in enumerate(config.var)
                offset = var.offset
                for pos = 1:config.dof[i][vi]
                    Dist.accumulate!(var, pos + offset, wf2)
                end
            end
        end

        if ne % measurefreq == 0 && ne >= neval / 100
            ##############################################################################
            for i in 1:N
                config.visited[i] += abs(weights[i] * padding_probability[i] * config.reweight[i]) / probability
                if isnothing(measure)
                    config.observable[i] += weights[i] * padding_probability[i] / probability
                else
                    relativeWeights[i] = weights[i] * padding_probability[i] / probability
                end
            end
            if isnothing(measure) == false
                (fieldcount(V) == 1) ?
                measure(config.var[1], config.observable, relativeWeights, config) :
                measure(config.var, config.observable, relativeWeights, config)
            end

            config.normalization += 1.0 * padding_probability[config.norm] / probability
            config.visited[config.norm] += config.reweight[config.norm] * padding_probability[config.norm] / probability
            # push!(kwargs["mem"], (1.0 / config.probability, config.weights[i] / config.probability))
        end
        if ne % 1000 == 0
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
