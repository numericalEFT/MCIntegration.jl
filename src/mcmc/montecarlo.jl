function markovchain_montecarlo(config::Configuration, integrand::Function, neval, print, save, timer; measurefreq=2, measure::Function=simple_measure, kwargs...)
    ##############  initialization  ################################
    # don't forget to initialize the diagram weight

    for var in config.var
        Dist.initialize!(var, config)
    end

    weight = integrand(config)
    setweight!(config, weight)
    config.absWeight = abs(integrand(config))


    # updates = [changeIntegrand,] # TODO: sample changeVariable more often
    # updates = [changeIntegrand, swapVariable,] # TODO: sample changeVariable more often
    # updates = [changeIntegrand, swapVariable, changeVariable] # TODO: sample changeVariable more often
    updates = [swapVariable, changeVariable] # TODO: sample changeVariable more often
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
            prop = 1.0
            for (vi, var) in enumerate(config.var)
                offset = var.offset
                for pos = 1:config.dof[config.curr][vi]
                    # Dist.accumulate!(var, pos + offset, 1.0)
                    # Dist.accumulate!(var, pos + offset, config.absWeight)
                    prop *= var.prop[pos+offset]
                end
            end
            for (vi, var) in enumerate(config.var)
                offset = var.offset
                for pos = 1:config.dof[config.curr][vi]
                    # need to make sure Δxᵢ*∫_xᵢ^xᵢ₊₁ dx f^2(x)dx is a constant
                    # where  Δxᵢ ∝ 1/prop ∝ Jacobian for the vegas map
                    # since the current weight is sampled with the probability density ∝ |f(x)|*reweight
                    # the estimator ∝ Δxᵢ*f^2(x)/(|f(x)|*reweight) = |f(x)|/prop/reweight

                    Dist.accumulate!(var, pos + offset, config.absWeight / prop / config.reweight[config.curr])
                    # Dist.accumulate!(var, pos + offset, config.absWeight)
                end
            end
            ##############################################################################

            measure(config)
            config.normalization += prop / config.absWeight / config.reweight[config.curr]
            # push!(kwargs[:mem], (config.var[1][1], prop, config.absWeight))
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

function simple_measure(config)
    if (config.observable isa AbstractVector) && (eltype(config.observable) <: Number)
        config.observable[config.curr] += config.relativeWeight
    elseif config.observable isa Number
        config.observable += config.relativeWeight
    else
        error("simple_measure only works with observable of the AbstractVector of Number or Number types!")
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
