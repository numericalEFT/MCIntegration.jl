function montecarlo(config::Configuration, integrand::Function, neval, print, save, timer; kwargs...)
    ##############  initialization  ################################
    # don't forget to initialize the diagram weight

    for i in 1:10000
        for var in config.var
            Dist.initialize!(var, config)
        end

        weights = integrand(config)
        config.probability = abs(weights[config.curr]) / Dist.probability(config, config.curr) * config.reweight[config.curr]
        config.weights = weights
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

        curr = 1 # only calculate the first the integral
        currdof = config.dof[curr]
        prop = 1.0
        for vi in eachindex(currdof)
            (currdof[vi] <= 0) && continue # return if the var has zero degree of freedom
            var = config.var[vi]
            for idx in 1:currdof[vi]
                prop *= Dist.create!(var, idx + var.offset, config)
            end
        end
        # sampler may want to reject, then prop has already been set to zero
        weights = integrand(config)
        config.observable += weights * prop
        config.normalization += 1.0 #should be 1!
        # push!(mem, weight * prop)

        ######## accumulate variable #################
        for (vi, var) in enumerate(config.var)
            offset = var.offset
            for pos = 1:config.dof[curr][vi]
                Dist.accumulate!(var, pos + offset, (abs(weights[config.curr])^2 * prop^2))
                # Dist.accumulate!(var, pos + offset, (abs(weight) * prop))
                # Dist.accumulate!(var, pos + offset, 1.0)
            end
        end
        ###############################################
    end

    return config
end