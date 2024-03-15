"""
    function montecarlo(config::Configuration{N,V,P,O,T}, integrand::Function, neval,
        verbose=0, debug=false;
        measurefreq::Int=1, measure::Union{Nothing,Function}=nothing) where {N,V,P,O,T}

This algorithm combines Vegas with Markov-chain Monte Carlo.
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

NOTE: If there are more than one integrals,  all of them are sampled and measured at each Markov-chain Monte Carlo step!

This algorithm is as efficient as the Vegas algorithm for low-dimensional integration, and
tends to be more robust than the Vegas algorithm for high-dimensional integration.

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
The following command calls the MC Vegas solver,
```julia-repl
julia> integrate((x, c)->(x[1]^2+x[2]^2); var = Continuous(0.0, 1.0), dof = [[2,],], verbose=-1, solver=:vegasmc)
Integral 1 = 0.6640840471808533 ± 0.000916060916265263   (reduced chi2 = 0.945)
```
"""

"""
    function montecarlo(config::Configuration{N,V,P,O,T}, integrand::Function, neval,
        verbose=0, debug=false;
        measurefreq::Int=1, measure::Union{Nothing,Function}=nothing) where {N,V,P,O,T}

This function applies a Markov-chain Monte Carlo (MCMC) technique combined with the Vegas algorithm to compute integrals. In addition to calculating the original integrals, it also introduces a normalization integral with an integrand ~ 1, which enhances the efficiency and robustness of high-dimensional integration tasks.

# Overview
Given multiple integrands involving multiple variables, the algorithm finds the best distribution ansatz that fits all integrands together. For instance, consider we want to calculate two integrals: ``f_1(x)`` and ``f_2(x, y)``, where ``x`` and ``y`` are two different types of variables. The algorithm learns distributions ``\\rho_x(x)`` and ``\\rho_y(y)`` such that ``f_1(x)/\\rho_x(x)`` and ``f_2(x, y)/\\rho_x(x)/\\rho_y(y)`` are as flat as possible.

Then, it samples variables ``x`` and ``y`` using the Metropolis-Hastings algorithm with a joint distribution `p(x, y)`,
```math
p(x, y) = r_0 \\cdot \\rho_x(x) \\cdot \\rho_y(y) + r_1 \\cdot |f_1(x)| \\cdot \\rho_y(y) + r_2 \\cdot |f_2(x, y)|
```
where ``r_i`` are certain reweighting factor to make sure all terms contribute same weights.

One can then estimate the integrals by averaging the observables ``f_1(x)\\rho_y(y)/p(x, y)`` and ``f_2(x, y)/p(x, y)``.

The algorithm defaults to the standard Vegas algorithm if ``r_0 = 1`` and ``r_{i>0} = 0``.

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
The following command calls the VegasMC solver,
```julia-repl
julia> integrate((x, c)->(x[1]^2+x[2]^2); var = Continuous(0.0, 1.0), dof = [[2,],], verbose=-1, solver=:vegasmc)
Integral 1 = 0.6640840471808533 ± 0.000916060916265263   (reduced chi2 = 0.945)
```
"""
function montecarlo(config::Configuration{N,V,P,O,T}, integrand::Function, neval,
    verbose=0, timer=[], debug=false;
    measure::Union{Nothing,Function}=nothing, measurefreq::Int=1, inplace::Bool=false
) where {N,V,P,O,T}

    @assert measurefreq > 0

    if isnothing(measure)
        @assert (config.observable isa AbstractVector) && (length(config.observable) == config.N) && (eltype(config.observable) == T) "the default measure can only handle observable as Vector{$T} with $(config.N) elements!"
    end
    weights = zeros(T, N)
    _weights = zeros(T, N)

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

    relativeWeights = zeros(T, N)
    # padding probability for user and normalization integrands
    # should be something like [f_1(x)*g_0(y), f_2(x, y), 1*f_0(x)*g_0(y)], where f_0(x) and g_0(y) are the ansatz from the Vegas map
    # after padding, all integrands have the same dimension and have similiar probability distribution
    padding_probability = zeros(Float64, N + 1)
    padding_probability_cache = zeros(Float64, N + 1) #used for cache the padding probability of the proposal configuration
    ##############  initialization  ################################
    # don't forget to initialize the diagram weight
    for var in config.var
        Dist.initialize!(var, config)
    end

    if inplace
        integrand((isone(length(config.var)) ? config.var[1] : config.var), _weights, config)
    else
        _weights .= integrand((isone(length(config.var)) ? config.var[1] : config.var), config)
    end

    padding_probability .= [Dist.padding_probability(config, i) for i in 1:N+1]
    probability = config.reweight[config.norm] * padding_probability[config.norm] #normalization integral
    for i in 1:config.N #other integrals
        weights[i] = _weights[i]
        probability += abs(_weights[i]) * config.reweight[i] * padding_probability[i]
    end
    # config.probability = newProbability


    updates = [changeVariable,]
    # for i = 2:length(config.var)*2
    #     push!(updates, changeVariable) #add other updates
    # end
    if debug
        for _update in updates
            MCUtility.test_type_stability(_update, (config, integrand, inplace, probability,
                weights, _weights, padding_probability, padding_probability_cache))
        end
    end

    ########### MC simulation ##################################
    startTime = time()

    for ne = 1:neval
        # config.neval += 1
        _update = rand(config.rng, updates) # randomly select an update
        probability = _update(config, integrand, inplace, probability,
            weights, _weights, padding_probability, padding_probability_cache)
        if debug && (isfinite(probability) == false)
            @warn("integrand probability = $(probability) is not finite at step $(neval)")
        end
        # WARNING: Don't turn it on, because some integral may actually vanish (for example, circle are)
        # if debug && (all(x -> isfinite(x), weights))
        #     @warn("integrand = $(weights) is not all finite at step $(neval)")
        # end

        ######## accumulate variable and calculate variable probability #################
        for i in 1:N
            # need to make sure Δxᵢ*∫_xᵢ^xᵢ₊₁ dx f^2(x)dx is a constant
            # where  Δxᵢ ∝ 1/prop ∝ Jacobian for the vegas map
            # since the current weight is sampled with the probability density ∝ config.probability =\sum_i |f_i(x)|*reweight[i]
            # the estimator ∝ Δxᵢ*f^2(x)/(|f(x)|*reweight) = |f(x)|/prop/reweight
            f2 = abs(weights[i])^2 / Dist.probability(config, i)
            wf2 = f2 * padding_probability[i] / probability
            for (vi, var) in enumerate(config.var)
                offset = var.offset
                for pos = 1:config.dof[i][vi]
                    Dist.accumulate!(var, pos + offset, wf2)
                end
            end
        end

        if ne % measurefreq == 0 && ne >= neval / 100
            ##############################################################################
            for i in 1:N
                config.visited[i] += abs(weights[i] * padding_probability[i] * config.reweight[i]) / probability
                if isnothing(measure)
                    config.observable[i] += weights[i] * padding_probability[i] / probability
                else
                    relativeWeights[i] = weights[i] * padding_probability[i] / probability
                end
            end
            if isnothing(measure) == false
                (fieldcount(V) == 1) ?
                measure(config.var[1], config.observable, weights, relativeWeights, config) :
                measure(config.var, config.observable, weights, relativeWeights, config)
                # (fieldcount(V) == 1) ?
                # measure(config.var[1], config.observable, relativeWeights, config) :
                # measure(config.var, config.observable, relativeWeights, config)
            end

            config.normalization += 1.0 * padding_probability[config.norm] / probability
            config.visited[config.norm] += config.reweight[config.norm] * padding_probability[config.norm] / probability
            # push!(kwargs["mem"], (1.0 / config.probability, config.weights[i] / config.probability))
        end
        if ne % 1000 == 0
            for t in timer
                check(t, config, neval)
            end
        end
    end

    return config
end

@inline function integrand_wrap(config, _integrand)
    return _integrand(config.var..., config)
end
