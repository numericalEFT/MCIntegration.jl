type_length(tup::Type{T}) where {T<:Tuple} = length(tup.parameters)

#return a function call: `integrand(config.var[1], config.var[2], ...config.var[N], config).
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

# @generated function sub2ind_gen(dims::NTuple{N}, I::Integer...) where {N}
#     ex = :(I[$N] - 1)
#     for i = (N-1):-1:1
#         ex = :(I[$i] - 1 + dims[$i] * $ex)
#     end
#     return :($ex + 1)
# end

"""
    function montecarlo(config::Configuration{Ni,V,P,O,T}, integrand::Function, neval,
        verbose=0, timer=[], debug=false;
        measure::Union{Nothing,Function}=nothing, measurefreq::Int=1) where {Ni,V,P,O,T}

This function implements the Vegas algorithm, a Monte Carlo method specifically designed for multi-dimensional integration. The underlying principle and methodology of the algorithm can be explored further in the [Vegas documentation](https://vegas.readthedocs.io/en/latest/background.html#importance-sampling).

# Overview
The Vegas algorithm employs an importance sampling scheme. For a one-dimensional integral with the integrand ``f(x)``, the algorithm constructs an optimized distribution ``\\rho(x)`` that approximates the integrand as closely as possible (a strategy known as the Vegas map trick; refer to [`Dist.Continuous`](@ref) for more details).

The variable ``x`` is then sampled using the distribution ``\\rho(x)``, and the integral is estimated by averaging the estimator ``f(x)/\\rho(x)``.

# Note
- If there are multiple integrals, all of them are sampled and measured at each Monte Carlo step.
- This algorithm is particularly efficient for low-dimensional integrations but might be less efficient and robust than the Markov-chain Monte Carlo solvers for high-dimensional integrations.


# Arguments
- `integrand`: A user-defined function evaluating the integrand. The function should be either `integrand(var, config)` or `integrand(var, weights, config)` depending on whether `inplace` is `false` or `true` respectively. Here, `var` are the random variables and `weights` is an output array to store the calculated weights. The last parameter passes the MC `Configuration` struct to the integrand, so that user has access to userdata, etc.

- `measure`: An optional user-defined function to accumulate the integrand weights into the observable. The function signature should be `measure(var, obs, relative_weights, config)`. Here, `obs` is a vector of observable values for each component of the integrand and `relative_weights` are the weights calculated from the integrand multiplied by the probability of the corresponding variables.

The following are the snippets of the `integrand` and `measure` functions:
```julia
function integrand(var, config)
    # calculate your integrand values
    # return integrand1, integrand2, ...
end
```
```julia
function measure(var, obs, weights, config)
    # accumulates the weight into the observable
    # For example,
    # obs[1] = weights[1] # integral 1
    # obs[2] = weights[2] # integral 2
    # ...
end
```

# Examples
The following command calls the Vegas solver,
```julia-repl
julia> integrate((x, c)->(x[1]^2+x[2]^2); var = Continuous(0.0, 1.0), dof = [[2,],], verbose=-1, solver=:vegas)
Integral 1 = 0.667203631824444 ± 0.0005046485925614018   (reduced chi2 = 1.46)
```
"""
function montecarlo(config::Configuration{Ni,V,P,O,T}, integrand::Function, neval,
    verbose=0, timer=[], debug=false;
    measure::Union{Nothing,Function}=nothing, measurefreq::Int=1, inplace::Bool=false
) where {Ni,V,P,O,T}

    @assert measurefreq > 0

    relativeWeights = zeros(T, Ni)
    weights = zeros(T, Ni)
    padding_probability = ones(Ni)
    diff = [config.dof[i] == config.maxdof for i in 1:Ni] # check if the dof is the same as the maxdof, if the same, then there is no need to update the padding probability

    ################## test integrand type stability ######################
    if debug
        if inplace
            if (length(config.var) == 1)
                MCUtility.test_type_stability(integrand, (config.var[1], weights, config))
            else
                MCUtility.test_type_stability(integrand, (config.var, weights, config))
            end
        else
            if (length(config.var) == 1)
                MCUtility.test_type_stability(integrand, (config.var[1], config))
            else
                MCUtility.test_type_stability(integrand, (config.var, config))
            end
        end
    end
    #######################################################################


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
        # Dist.padding_probability!(config, padding_probability)
        for i in 1:Ni
            if diff[i] == false
                padding_probability[i] = Dist.padding_probability(config, i)
            end
        end
        # weights = @_expanded_integrand(config, integrand, 1) # very fast, but requires explicit N
        # weights = integrand_wrap(config, integrand) #make a lot of allocations
        if inplace
            integrand((isone(fieldcount(V)) ? config.var[1] : config.var), weights, config)
        else
            weights .= integrand((isone(fieldcount(V)) ? config.var[1] : config.var), config)
        end

        # println("before: ", weights, "with jac = ", jac)

        if (ne % measurefreq == 0)
            if isnothing(measure)
                # println("after: ", weights * jac)
                for i in 1:Ni
                    config.observable[i] += weights[i] * padding_probability[i] * jac
                end
                # observable += weights * jac
            else
                for i in 1:Ni
                    relativeWeights[i] = weights[i] * padding_probability[i] * jac
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
                # ! warning: need to check whether to use jac or jac*padding_probability[idx]
                if debug && (isfinite(w2) == false)
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
