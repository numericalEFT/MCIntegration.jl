function montecarlo(config::Configuration, integrand::Function, neval, print, save, timer; kwargs...)
    ##############  initialization  ################################
    # don't forget to initialize the diagram weight
    weight = integrand(config)
    setweight!(config, weight)
    config.absWeight = abs(integrand(config))

    ########### MC simulation ##################################
    startTime = time()

    for i = 1:neval
        config.neval += 1

        curr = 1 # only calculate the first the integral
        currdof = config.dof[curr]
        prop = 1.0
        for vi in eachindex(currdof)
            (currdof[vi] <= 0) && continue # return if the var has zero degree of freedom
            var = config.var[vi]
            for idx in 1:currdof[vi]
                prop *= Dist.shift!(var, idx + var.offset, config)
            end
        end
        # sampler may want to reject, then prop has already been set to zero
        weight = integrand(config)
        config.obs .+= weight .* prop

        ######## accumulate variable #################
        for (vi, var) in enumerate(config.var)
            offset = var.offset
            for pos = 1:config.dof[curr][vi]
                Dist.accumulate!(var, pos + offset)
            end
        end
        ###############################################
    end

    return config
end