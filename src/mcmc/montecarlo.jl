mutable struct _State{T}
    weight::T
    probability::Float64
end

function montecarlo(config::Configuration{N,V,P,O,T}, integrand::Function, neval,
    print=0, save=0, timer=[], debug=false;
    measurefreq=2, measure::Union{Nothing,Function}=nothing, kwargs...) where {N,V,P,O,T}
    ##############  initialization  ################################
    # don't forget to initialize the diagram weight
    if isnothing(measure)
        @assert (config.observable isa AbstractVector) && (length(config.observable) == config.N) && (eltype(config.observable) == T) "the default measure can only handle observable as Vector{$T} with $(config.N) elements!"
    end

    ################## test integrand type stability ######################
    if debug
        if (length(config.var) == 1)
            MCUtility.test_type_stability(integrand, (1, config.var[1], config))
        else
            MCUtility.test_type_stability(integrand, (1, config.var, config))
        end
    end
    #######################################################################

    state = _State{T}(zero(T), 1.0)

    for i in 1:10000
        initialize!(config, integrand, state)
        if (config.curr == config.norm) || state.probability > TINY
            break
        end
    end
    if (config.curr != config.norm) && state.probability ≈ 0.0
        error("Cannot find the variables that makes the $(config.curr) integrand nonzero!")
    elseif (config.curr != config.norm) && state.probability < TINY
        @warn("Cannot find the variables that makes the $(config.curr) integrand >1e-10!")
    end

    # updates = [changeIntegrand,] # TODO: sample changeVariable more often
    # updates = [changeIntegrand, swapVariable,] # TODO: sample changeVariable more often
    updates = [changeIntegrand, swapVariable, changeVariable] # TODO: sample changeVariable more often
    for i = 2:length(config.var)*2
        push!(updates, changeVariable)
    end

    if debug
        for _update in updates
            MCUtility.test_type_stability(_update, (config, integrand, state))
        end
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
        _update(config, integrand, state)
        # push!(kwargs[:mem], (config.curr, config.relativeWeight))
        # if i % 10 == 0 && i >= neval / 100
        if debug && (isfinite(state.probability) == false)
            @warn("integrand probability = $(state.probability) is not finite at step $(config.neval)")
        end
        if i % measurefreq == 0 && i >= neval / 100

            ######## accumulate variable #################
            if config.curr != config.norm
                for (vi, var) in enumerate(config.var)
                    offset = var.offset
                    for pos = 1:config.dof[config.curr][vi]
                        Dist.accumulate!(var, pos + offset, 1.0)
                    end
                end
            end
            ###############################################

            if config.curr == config.norm # the last diagram is for normalization
                config.normalization += 1.0 / config.reweight[config.norm]
            else
                curr = config.curr
                relativeWeight = state.weight / state.probability
                if isnothing(measure)
                    config.observable[curr] += relativeWeight
                else
                    (fieldcount(V) == 1) ?
                    measure(config.curr, config.var[1], config.observable, relativeWeight, config) :
                    measure(config.curr, config.var, config.observable, relativeWeight, config)
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

@inline function integrand_wrap(new, config, _integrand)
    return _integrand(new, config.var..., config)
end

function initialize!(config::Configuration{N,V,P,O,T}, integrand, state) where {N,V,P,O,T}
    for var in config.var
        Dist.initialize!(var, config)
    end
    curr = config.curr
    if curr != config.norm
        # config.weights[curr] = integrand_wrap(curr, config, integrand)
        state.weight = (length(config.var) == 1) ? integrand(curr, config.var[1], config) : integrand(curr, config.var, config)
        state.probability = abs(state.weight) * config.reweight[curr]
    else
        state.weight = zero(T)
        state.probability = config.reweight[curr]
    end
    return
end

# function setweight!(config, weight)
#     config.relativeWeight = weight / abs(weight) / config.reweight[config.curr]
# end


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