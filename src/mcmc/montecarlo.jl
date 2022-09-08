mutable struct _State{T}
    curr::Int
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

    if haskey(kwargs, :idx)
        curr = kwargs[:idx]
    elseif haskey(kwargs, :curr)
        curr = kwargs[:curr]
    else
        curr = 1
    end

    state = _State{T}(curr, zero(T), 1.0)

    for i in 1:10000
        initialize!(config, integrand, state)
        if (state.curr == config.norm) || state.probability > TINY
            break
        end
    end
    if (state.curr != config.norm) && state.probability ≈ 0.0
        error("Cannot find the variables that makes the $(state.curr) integrand nonzero!")
    elseif (state.curr != config.norm) && state.probability < TINY
        @warn("Cannot find the variables that makes the $(state.curr) integrand >1e-10!")
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
        config.visited[state.curr] += 1
        _update = rand(config.rng, updates) # randomly select an update
        _update(config, integrand, state)
        # push!(kwargs[:mem], (config.curr, config.relativeWeight))
        # if i % 10 == 0 && i >= neval / 100
        if debug && (isfinite(state.probability) == false)
            @warn("integrand probability = $(state.probability) is not finite at step $(config.neval)")
        end
        if i % measurefreq == 0 && i >= neval / 100

            ######## accumulate variable #################
            if state.curr != config.norm
                for (vi, var) in enumerate(config.var)
                    offset = var.offset
                    for pos = 1:config.dof[state.curr][vi]
                        Dist.accumulate!(var, pos + offset, 1.0)
                    end
                end
            end
            ###############################################

            if state.curr == config.norm # the last diagram is for normalization
                config.normalization += 1.0 / config.reweight[config.norm]
            else
                curr = state.curr
                relativeWeight = state.weight / state.probability
                if isnothing(measure)
                    config.observable[curr] += relativeWeight
                else
                    (fieldcount(V) == 1) ?
                    measure(state.curr, config.var[1], config.observable, relativeWeight, config) :
                    measure(state.curr, config.var, config.observable, relativeWeight, config)
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
    curr = state.curr
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