function montecarlo(config::Configuration{N,V,P,O,T}, integrand::Function, neval,
    print, save, timer;
    measurefreq=2, measure::Union{Nothing,Function}=nothing, kwargs...) where {N,V,P,O,T}
    ##############  initialization  ################################
    # don't forget to initialize the diagram weight
    if isnothing(measure)
        @assert (config.observable isa AbstractVector) && (length(config.observable) == config.N) && (eltype(config.observable) == T) "the default measure can only handle observable as Vector{$T} with $(config.N) elements!"
    end

    for i in 1:10000
        initialize!(config, integrand)
        if (config.curr == config.norm) || config.probability > TINY
            break
        end
    end
    @assert (config.curr == config.norm) || config.probability > TINY "Cannot find the variables that makes the $(config.curr) integrand >1e-10"

    # updates = [changeIntegrand,] # TODO: sample changeVariable more often
    # updates = [changeIntegrand, swapVariable,] # TODO: sample changeVariable more often
    updates = [changeIntegrand, swapVariable, changeVariable] # TODO: sample changeVariable more often
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
        # push!(kwargs[:mem], (config.curr, config.relativeWeight))
        # if i % 10 == 0 && i >= neval / 100
        if isfinite(config.probability) == false
            @warn("integrand probability = $(config.probability) is not finite at step $(config.neval)")
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
                relativeWeight = config.weights[curr] / config.probability
                if isnothing(measure)
                    config.observable[curr] += relativeWeight
                else
                    measure(config.curr, config.observable, relativeWeight, config)
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

function initialize!(config, integrand)
    for var in config.var
        Dist.initialize!(var, config)
    end
    curr = config.curr
    if curr != config.norm
        config.weights[curr] = integrand_wrap(curr, config, integrand)
        config.probability = abs(config.weights[curr]) * config.reweight[curr]
    else
        config.probability = config.reweight[curr]
    end

    # setweight!(config, weight)
    # config.absWeight = abs(weight)
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