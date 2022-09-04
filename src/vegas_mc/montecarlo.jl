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

function montecarlo(config::Configuration{V,P,O,T}, integrand::Function, neval,
    print, save, timer;
    measurefreq=2, measure::Union{Nothing,Function}=nothing,
    kwargs...) where {V,P,O,T}

    if isnothing(measure)
        @assert (config.observable isa AbstractVector) && (length(config.observable) == config.N) && (eltype(config.observable) == T) "the default measure can only handle observable as Vector{$T} with $(config.N) elements!"
    end
    ##############  initialization  ################################
    # don't forget to initialize the diagram weight
    for i in 1:10000
        initialize!(config, integrand)
        if (config.curr == config.norm) || abs(config.weights[config.curr]) > TINY
            break
        end
    end
    @assert (config.curr == config.norm) || abs(config.weights[config.curr]) > TINY "Cannot find the variables that makes the $(config.curr) integrand >1e-10"


    # updates = [changeIntegrand,] # TODO: sample changeVariable more often
    # updates = [changeIntegrand, swapVariable,] # TODO: sample changeVariable more often
    # updates = [changeIntegrand, swapVariable, changeVariable] # TODO: sample changeVariable more often
    updates = [changeIntegrand, changeVariable] # TODO: sample changeVariable more often
    # updates = [changeVariable,] # TODO: sample changeVariable more often
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
        # if i % 10 == 0 && i >= neval / 100
        if i % measurefreq == 0 && i >= neval / 100

            ######## accumulate variable and calculate variable probability #################
            if config.curr != config.norm

                prop = Dist.probability(config, config.curr)
                f2 = (abs(config.weights[config.curr]))^2 / prop

                for (vi, var) in enumerate(config.var)
                    offset = var.offset
                    for pos = 1:config.dof[config.curr][vi]
                        # need to make sure Δxᵢ*∫_xᵢ^xᵢ₊₁ dx f^2(x)dx is a constant
                        # where  Δxᵢ ∝ 1/prop ∝ Jacobian for the vegas map
                        # since the current weight is sampled with the probability density ∝ |f(x)|*reweight
                        # the estimator ∝ Δxᵢ*f^2(x)/(|f(x)|*reweight) = |f(x)|/prop/reweight
                        # Dist.accumulate!(var, pos + offset, f2 / config.probability)

                        # the following accumulator has a similar performance
                        # this is because that with an optimial grid, |f(x)| ~ prop
                        # Dist.accumulate!(var, pos + offset, 1.0/ reweight
                        Dist.accumulate!(var, pos + offset, 1.0 / config.reweight[config.curr])
                    end
                end
            end
            ##############################################################################

            if isnothing(measure)
                for i in eachindex(config.weights)
                    prob = Dist.delta_probability(config, config.curr; new=i)
                    config.observable[i] += config.weights[i] * prob / config.probability
                end
            else
                for i in eachindex(config.weights)
                    prob = Dist.delta_probability(config, config.curr; new=i)
                    config.relativeWeights[i] = config.weights[i] * prob / config.probability
                end
                measure(config.observable, config.relativeWeights, config)
            end
            prob = Dist.delta_probability(config, config.curr; new=config.norm)
            config.normalization += prob / config.probability
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

    weights = integrand_wrap(config, integrand)
    # config.probability = abs(weights[config.curr]) / Dist.probability(config, config.curr) * config.reweight[config.curr]
    if config.curr == config.norm
        config.probability = config.reweight[config.curr]
    else
        # config.probability = abs(weights[config.curr]) * config.reweight[config.curr] / Dist.probability(config, config.curr)
        config.probability = abs(weights[config.curr]) * config.reweight[config.curr]
    end
    setWeight!(config, weights)
end

function setWeight!(config, weights)
    # weights can be a number of a vector of numbers
    # this function copy weights to a vector config.weights
    for i in eachindex(config.weights)
        config.weights[i] = weights[i]
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
