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
        print=0, save=0, timer=[], debug=false;
        measure::Union{Nothing,Function}=nothing, measurefreq::Int=1) where {Ni,V,P,O,T}

This algorithm implements the classic Vegas algorithm.

The main idea of the algorithm can be found in this [link](https://vegas.readthedocs.io/en/latest/background.html#importance-sampling).

The algorithm uses a simple important sampling scheme. 
Consider an one-dimensional integral with the integrand ``f(x)``, the algorithm will try to learn an
optimized distribution ``\\rho(x)>0`` which mimic the integrand as good as possible (a.k.a, the Vegas map trick, see [`Dist.Continuous`](@ref)) for more detail. 

One then sample the variable ``x`` with the distribution ``\\rho(x)``, and estimate the integral by averging the estimator ``f(x)/\\rho(x)``.

NOTE: If there are more than one integrals, then all integrals are sampled and measured at each Monte Carlo step.

This algorithm is very efficient for low-dimensional integrations, but can be less
efficient and less robust than the Markov-chain Monte Carlo solvers for high-dimensional integrations.

# Arguments
- `integrand` : User-defined function with the following signature:
```julia
function integrand(var, config)
    # calculate your integrand values
    # return integrand1, integrand2, ...
end
```
The first parameter `var` is either a Variable struct if there is only one type of variable, or a tuple of Varibles if there are more than one types of variables.
The second parameter passes the MC `Configuration` struct to the integrand, so that user has access to userdata, etc.

- `measure` : User-defined function with the following signature:
```julia
function measure(var, obs, weights, config)
    # accumulates the weight into the observable
    # For example,
    # obs[1] = weights[1] # integral 1
    # obs[2] = weights[2] # integral 2
    # ...
end
```
The first argument `var` is either a Variable struct if there is only one type of variable, or a tuple of Varibles if there are more than one types of variables.
The second argument passes the user-defined observable to the function, it should be a vector with the length same as the integral number.
The third argument is the integrand weights to be accumulated to the observable, it is a vector with the length same as the integral number.
The last argument passes the MC `Configuration` struct to the integrand, so that user has access to userdata, etc.

# Examples
The following command calls the Vegas solver,
```julia-repl
julia> integrate((x, c)->(x[1]^2+x[2]^2); var = Continuous(0.0, 1.0), dof = 2, print=-1, solver=:vegas)
Integral 1 = 0.667203631824444 ± 0.0005046485925614018   (chi2/dof = 1.46)
```
"""
function montecarlo(config::Configuration{Ni,V,P,O,T}, integrand::Function, neval,
    print=0, save=0, timer=[], debug=false;
    measure::Union{Nothing,Function}=nothing, measurefreq::Int=1
) where {Ni,V,P,O,T}

    @assert measurefreq > 0

    relativeWeights = zeros(T, Ni)
    weights = zeros(T, Ni)

    ################## test integrand type stability ######################
    if debug
        if (length(config.var) == 1)
            MCUtility.test_type_stability(integrand, (config.var[1], config))
        else
            MCUtility.test_type_stability(integrand, (config.var, config))
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
