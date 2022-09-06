type_length(tup::Type{T}) where {T<:Tuple} = length(tup.parameters)

"""
    macro _expanded_integrand(config, integrand, N)
    
    return a function call: `integrand(config.var[1], config.var[2], ...config.var[N], config).
"""
macro _expanded_integrand(config, integrand, N)
    #TODO: right now, it only works for explict N
    para = []
    # NN = type_length($(esc(V)))
    # println(NN)
    for i = 1:N
        push!(para, :($(esc(config)).var[$i]))
    end
    return Expr(:call, :($(esc(integrand))), para..., :($(esc(config))))
end

# @generated function _gen_integrand(config::Configuration{Ni,V,P,O,T}, integrand) where {Ni,V,P,O,T}
# end

@generated function sub2ind_gen(dims::NTuple{N}, I::Integer...) where {N}
    ex = :(I[$N] - 1)
    for i = (N-1):-1:1
        ex = :(I[$i] - 1 + dims[$i] * $ex)
    end
    return :($ex + 1)
end

function montecarlo(config::Configuration{Ni,V,P,O,T}, integrand::Function, neval,
    print=0, save=0, timer=[];
    measure::Union{Nothing,Function}=nothing, measurefreq=1, kwargs...) where {Ni,V,P,O,T}

    relativeWeights = zeros(T, Ni)

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

    for ne = 1:neval
        config.neval += 1

        maxdof = config.maxdof
        jac = 1.0
        for vi in eachindex(maxdof)
            var = config.var[vi]
            for idx in 1:maxdof[vi]
                Dist.shift!(var, idx + var.offset, config)
                jac /= var.prob[idx+var.offset]

                # alternative way to calculate the jacobian
                # jac *= Dist.create!(var, idx + var.offset, config)
            end
        end
        # weights = @_expanded_integrand(config, integrand, 1) # very fast, but requires explicit N
        # weights = integrand_wrap(config, integrand) #make a lot of allocations
        weights = (fieldcount(V) == 1) ? integrand(config.var[1], config) : integrand(config.var, config)

        if (ne % measurefreq == 0)
            if isnothing(measure)
                for i in 1:Ni
                    config.observable[i] += weights[i] * jac
                end
                # observable += weights * jac
            else
                for i in 1:Ni
                    relativeWeights[i] = weights[i] * jac
                end
                (fieldcount(V) == 1) ?
                measure(config.var[1], config.observable, relativeWeights, config) :
                measure(config.var, config.observable, relativeWeights, config)
            end
            # push!(mem, weight * prop)
            config.normalization += 1.0 #should be 1!
        end
        # w2 = abs(weights)
        # Dist.accumulate!(config.var[1], 1, (w2 * jac)^2)

        ######## accumulate variable #################
        for (vi, var) in enumerate(config.var)
            offset = var.offset
            for idx in eachindex(weights)
                w2 = abs(weights[idx])
                j2 = jac
                if isfinite(w2) == false
                    @warn("abs of the integrand $idx = $(w2) is not finite at step $(config.neval)")
                end
                for pos = 1:config.dof[idx][vi]
                    Dist.accumulate!(var, pos + offset, (w2 * j2)^2)
                    # Dist.accumulate!(var, pos + offset, (abs(weight) * prop))
                    # Dist.accumulate!(var, pos + offset, 1.0)
                end
            end
        end
        ###############################################
    end

    # config.observable[1] = observable
    return config
end

@inline function integrand_wrap(config::Configuration{N,V,P,O,T}, _integrand) where {N,V,P,O,T}
    # return _integrand(config.var..., config)
    if fieldcount(V) == 1
        return _integrand(config.var[1], config)
    else
        return _integrand(config.var, config)
    end
    # return _integrand(config)
end
