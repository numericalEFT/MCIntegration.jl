mutable struct _State{T}
    curr::Int
    weight::T
    probability::Float64
end

"""

    function montecarlo(config::Configuration{N,V,P,O,T}, integrand::Function, neval,
        verbose=0, timer=[], debug=false;
        measurefreq::Int=1, measure::Union{Nothing,Function}=nothing, idx::Int=1) where {N,V,P,O,T}

This algorithm computes high-dimensional integrals using a Markov-chain Monte Carlo (MCMC) method. It is particularly well-suited for cases involving multiple integrals over several variables. 

The MCMC algorithm learns an optimal distribution, or 'ansatz', to best fit all integrands under consideration. Additionally, it introduces a normalization integral (with an integrand ~ 1) alongside the original integrals.

Assume we have two integrals to compute, ``f_1(x)`` and ``f_2(x, y)``, where ``x`` and ``y`` are variables of different types. The algorithm aims to learn the distributions ``\\rho_x(x)`` and ``\\rho_y(y)``, such that the quantities ``f_1(x)/\\rho_x(x)`` and ``f_2(x, y)/\\rho_x(x)/\\rho_y(y)`` are as flat as possible.

Using the Metropolis-Hastings algorithm, the algorithm samples variables ``x`` and ``y`` based on the joint distribution:
```math
p(x, y) = r_0 \\cdot \\rho_x(x) \\cdot \\rho_y(y) + r_1 \\cdot |f_1(x)| \\cdot \\rho_y(y) + r_2 \\cdot |f_2(x, y)|
```
where ``r_i`` are reweighting factors ensuring equal contribution from all terms. The integrals are then estimated by averaging the observables ``f_1(x)\\rho_y(y)/p(x, y)`` and ``f_2(x, y)/p(x, y)``.

Setting ``r_0 = 1`` and ``r_{i>0} = 0`` reduces this algorithm to the classic Vegas algorithm.

The key difference between this MCMC method and the :vegasmc solver lies in how the joint distribution ``p(x, y)`` is sampled. This MCMC solver uses the Metropolis-Hastings algorithm to sample each term in ``p(x, y)`` as well as the variables ``(x, y)``. The MC configuration space thus consists of `(idx, x, y)`, where `idx` represents the index of the user-defined and normalization integrals. In contrast, the `:vegasmc` algorithm only samples the `(x, y)` space, explicitly calculating all terms in ``p(x, y)`` on-the-fly for a given set of ``x`` and ``y``.

Note: When multiple integrals are involved, only one of them is sampled and measured at each Markov-chain Monte Carlo step!

While the MCMC method may be less efficient than the `:vegasmc or `:vegas` solvers for low-dimensional integrations, it exhibits superior efficiency and robustness when faced with a large number of integrals, a scenario where the `:vegasmc` and `:vegas` solvers tend to struggle.

# Arguments
- `integrand`: A user-defined function evaluating the integrand. The function should be `integrand(idx, var, config)`. Here, `idx` is the index of the integrand component to be evaluated, `var` are the random variables and `weights` is an output array to store the calculated weights. The last parameter passes the MC `Configuration` struct to the integrand, so that user has access to userdata, etc.

- `measure`: An optional user-defined function to accumulate the integrand weights into the observable. The function signature should be `measure(idx, var, obs, relative_weight, config)`. Here, `idx` is the integrand index, `obs` is a vector of observable values for each component of the integrand and `relative_weight` is the weight calculated from the `idx`-th integrand multiplied by the probability of the corresponding variables. 

The following are the snippets of the `integrand` and `measure` functions:
```julia
function integrand(idx, var, config)
    # calculate your integrand values
    # return integrand of the index idx
end
```
```julia
function measure(idx, var, obs, relative_weight, config)
    # accumulates the weight into the observable
    # For example,
    # obs[idx] = relative_weight # integral idx
    # ...
end
```

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
julia> integrate((idx, x, c)->(x[1]^2+x[2]^2); var = Continuous(0.0, 1.0), dof = [[2,],], verbose=-1, solver=:mcmc)
Integral 1 = 0.6757665376867902 ± 0.008655534861083898   (reduced chi2 = 0.681)
```
"""
function montecarlo(config::Configuration{N,V,P,O,T}, integrand::Function, neval,
    verbose=0, timer=[], debug=false;
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
    # if debug
    #     if (length(config.var) == 1)
    #         MCUtility.test_type_stability(integrand, (1, config.var[1], config))
    #     else
    #         MCUtility.test_type_stability(integrand, (1, config.var, config))
    #     end
    # end
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
        error("Cannot find the variables that makes the #$(state.curr) integrand nonzero!")
    elseif (state.curr != config.norm) && state.probability < TINY
        @warn("Cannot find the variables that makes the #$(state.curr) integrand >1e-10!")
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
        # println(config.var[1][1], ", ", config.var[1][2], ", ", integrand(curr, config.var[1], config))
        state.probability = abs(state.weight) * config.reweight[curr]
    else
        state.weight = zero(T)
        state.probability = config.reweight[curr]
    end
    return
end