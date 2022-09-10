mutable struct _State{T}
    curr::Int
    weight::T
    probability::Float64
end

"""

    function montecarlo(config::Configuration{N,V,P,O,T}, integrand::Function, neval,
        print=0, save=0, timer=[], debug=false;
        measurefreq::Int=1, measure::Union{Nothing,Function}=nothing, idx::Int=1) where {N,V,P,O,T}

This algorithm calculate high-dimensional integrals with a Markov-chain Monte Carlo.
For multiple integrands invoves multiple variables, it finds the best distribution
ansatz to fit them all together. In additional to the original integral, it also 
introduces a normalization integral with integrand ~ 1.

Assume we want to calculate the integral ``f_1(x)`` and ``f_2(x, y)``, where x, y are two different types of variables.
The algorithm will try to learn a distribution ``\\rho_x(x)`` and ``\\rho_y(y)`` so that ``f_1(x)/\\rho_x(x)`` and ``f_2(x, y)/\\rho_x(x)/\\rho_y(y)``
are as flat as possible. 

The algorithm then samples the variables x and y with a joint distribution using the Metropolis-Hastings algorithm,
```math
p(x, y) = r_0 \\cdot \\rho_x(x) \\cdot \\rho_y(y) + r_1 \\cdot |f_1(x)| \\cdot \\rho_y(y) + r_2 \\cdot |f_2(x, y)|
```
where ``r_i`` are certain reweighting factor to make sure all terms contribute same weights.
One then estimate the integrals by averaging the observables ``f_1(x)\\rho_y(y)/p(x, y)`` and ``f_2(x, y)/p(x, y)``.

This algorithm reduces to the vanilla Vegas algorithm by setting ``r_0 = 1`` and ``r_{i>0} = 0``.

The key difference between this algorithmm and the algorithm `:vegasmc` is the way that the joint distribution ``p(x, y)`` is sampled.
In this algorithm, one use Metropolis-Hastings algorithm to sample each term in ``p(x, y)`` as well as the variables ``(x, y)``, so that the MC configuration space consists of
``(idx, x, y)``, where ``idx`` is the index of the user-defined and the normalization integrals. On the other hand, the `:vegasmc` algorithm
only uses Metropolis-Hastings algorithm to sample the configuration space ``(x, y)``. For a given set of x and y, all terms in ``p(x, y)`` are
explicitly calculated on the fly. If one can afford calculating all the integrands on the fly, then `:vegasmc` should be more efficient than this algorithm.

NOTE: If there are more than one integrals, only one of them are sampled and measured at each Markov-chain Monte Carlo step!

For low-dimensional integrations, this algorithm is much less efficient than the :vegasmc or :vegas solvers. For high-dimension integrations, however,
this algorithm becomes as efficent and robust as the :vegasmc solver, and is more efficient and robust than the :vegas solver.

# Arguments
- `integrand` : User-defined function with the following signature:
```julia
function integrand(idx, var, config)
    # calculate your integrand values
    # return integrand of the index idx
end
```
The first argument `idx` is index of the integral being sampled.
The second parameter `var` is either a Variable struct if there is only one type of variable, or a tuple of Varibles if there are more than one types of variables.
The third parameter passes the MC `Configuration` struct to the integrand, so that user has access to userdata, etc.

- `measure` : User-defined function with the following signature:
```julia
function measure(idx, var, obs, weight, config)
    # accumulates the weight into the observable
    # For example,
    # obs[idx] = weight # integral idx
    # ...
end
```
The first argument `idx` is index of the integral being sampled.
The second argument `var` is either a Variable struct if there is only one type of variable, or a tuple of Varibles if there are more than one types of variables.
The third argument passes the user-defined observable to the function, it should be a vector with the length same as the integral number.
The fourth argument is the integrand weights to be accumulated to the observable, it is a vector with the length same as the integral number.
The last argument passes the MC `Configuration` struct to the integrand, so that user has access to userdata, etc.

# Remark:

- What if the integral result makes no sense?

  One possible reason is the reweight factor. It is important for the Markov chain to visit the integrals with the similar frequency. 
  However, the weight of different integrals may be order-of-magnitude different. It is thus important to reweight the integrals. 
  Internally, the MC sampler try to reweight for each iteration. However, it could fail either 1) the total MC steps is too small so that 
  reweighting doesn't have enough time to show up; ii) the integrals are simply too different, and the internal reweighting subroutine is 
  not smart enough to figure out such difference. If 1) is the case, one either increase the neval. If 2) is the case, one may mannually 
  provide an array of reweight factors when initializes the `MCIntegration.configuration` struct. 

# Examples
The following command calls the MC Vegas solver,
```julia-repl
julia> integrate((idx, x, c)->(x[1]^2+x[2]^2); var = Continuous(0.0, 1.0), dof = 2, print=-1, solver=:mcmc)
Integral 1 = 0.6757665376867902 ± 0.008655534861083898   (chi2/dof = 0.681)
```
"""
function montecarlo(config::Configuration{N,V,P,O,T}, integrand::Function, neval,
    print=0, save=0, timer=[], debug=false;
    measurefreq::Int=1,
    measure::Union{Nothing,Function}=nothing,
    idx::Int=1 # the integral to start with
) where {N,V,P,O,T}

    @assert measurefreq > 0

    ##############  initialization  ################################
    # don't forget to initialize the diagram weight
    if isnothing(measure)
        @assert (config.observable isa AbstractVector) && (length(config.observable) == config.N) && (eltype(config.observable) == T) "the default measure can only handle observable as Vector{$T} with $(config.N) elements!"
    end
    # println("kwargs in mcmc: ", kwargs)

    ################## test integrand type stability ######################
    if debug
        if (length(config.var) == 1)
            MCUtility.test_type_stability(integrand, (1, config.var[1], config))
        else
            MCUtility.test_type_stability(integrand, (1, config.var, config))
        end
    end
    #######################################################################

    curr = idx

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
        # config.neval += 1
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