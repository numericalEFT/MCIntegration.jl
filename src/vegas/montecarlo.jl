function montecarlo(config::Configuration, integrand::Function,
    neval, print, save, timer;
    measure::Function=simple_measure, measurefreq=1, kwargs...)
    ##############  initialization  ################################
    # don't forget to initialize the diagram weight

    for i in 1:10000
        for var in config.var
            Dist.initialize!(var, config)
        end

        weights = integrand(config)
        config.probability = abs(weights[config.curr]) / Dist.probability(config, config.curr) * config.reweight[config.curr]
        for i in eachindex(config.weights)
            config.weights[i] = weights[i]
        end
        if abs(weights[config.curr]) > TINY
            break
        end
    end
    @assert abs(config.weights[config.curr]) > TINY "Cannot find the variables that makes the $(config.curr) integrand >1e-10"

    ########### MC simulation ##################################
    startTime = time()
    # mem = kwargs[:mem]

    for i = 1:neval
        config.neval += 1

        maxdof = config.maxdof
        prop = 1.0
        for vi in eachindex(maxdof)
            (maxdof[vi] <= 0) && continue # return if the var has zero degree of freedom
            var = config.var[vi]
            for idx in 1:maxdof[vi]
                prop *= Dist.create!(var, idx + var.offset, config)
            end
        end
        weights = integrand(config)
        for i in eachindex(config.weights)
            config.relativeWeight[i] = weights[i] * prop
        end

        if i % measurefreq == 0
            measure(config)
            config.normalization += 1.0 #should be 1!
            # push!(mem, weight * prop)
        end

        ######## accumulate variable #################
        for (vi, var) in enumerate(config.var)
            offset = var.offset
            for idx in eachindex(weights)
                for pos = 1:config.dof[idx][vi]
                    Dist.accumulate!(var, pos + offset, (abs(weights[idx])^2 * prop^2))
                    # Dist.accumulate!(var, pos + offset, (abs(weight) * prop))
                    # Dist.accumulate!(var, pos + offset, 1.0)
                end
            end
        end
        ###############################################
    end

    return config
end

function simple_measure(config)
    for i in eachindex(config.weights)
        config.observable[i] += config.relativeWeight[i]
    end
end