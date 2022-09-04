function montecarlo(config::Configuration{V,P,O,T}, integrand::Function, neval,
    print=0, save=0, timer=[];
    measure::Union{Nothing,Function}=nothing, measurefreq=1, kwargs...) where {V,P,O,T}

    if isnothing(measure)
        @assert (config.observable isa AbstractVector) && (length(config.observable) == config.N) && (eltype(config.observable) == T) "the default measure can only handle observable as Vector{$T} with $(config.N) elements!"
    end
    ##############  initialization  ################################
    # don't forget to initialize the variables 
    for var in config.var
        Dist.initialize!(var, config)
    end
    # Vegas doesn't need to initialize the weights

    ########### MC simulation ##################################
    startTime = time()
    # mem = kwargs[:mem]

    for i = 1:neval
        config.neval += 1

        maxdof = config.maxdof
        jac = 1.0
        for vi in eachindex(maxdof)
            var = config.var[vi]
            for idx in 1:maxdof[vi]
                jac *= Dist.create!(var, idx + var.offset, config)
            end
        end
        weights = integrand_wrap(config, integrand)

        if (i % measurefreq == 0)
            if isnothing(measure)
                for i in eachindex(weights)
                    config.observable[i] += weights[i] * jac
                end
            else
                for i in eachindex(weights)
                    config.relativeWeights[i] = weights[i] * jac
                end
                measure(config.observable, config.relativeWeights, config)
            end
            # push!(mem, weight * prop)
            config.normalization += 1.0 #should be 1!
        end

        ######## accumulate variable #################
        for (vi, var) in enumerate(config.var)
            offset = var.offset
            for idx in eachindex(weights)
                for pos = 1:config.dof[idx][vi]
                    Dist.accumulate!(var, pos + offset, (abs(weights[idx])^2 * jac^2))
                    # Dist.accumulate!(var, pos + offset, (abs(weight) * prop))
                    # Dist.accumulate!(var, pos + offset, 1.0)
                end
            end
        end
        ###############################################
    end

    return config
end

@inline function integrand_wrap(config, _integrand)
    return _integrand(config.var..., config)
end
