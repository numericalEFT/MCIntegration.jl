# MCIntegration

Robust and efficient Monte Carlo calculator for high-dimensional integral.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://numericalEFT.github.io/MCIntegration.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://numericalEFT.github.io/MCIntegration.jl/dev)
[![Build Status](https://github.com/numericalEFT/MCIntegration.jl/workflows/CI/badge.svg)](https://github.com/numericalEFT/MCIntegration.jl/actions)
[![Coverage](https://codecov.io/gh/numericalEFT/MCIntegration.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/numericalEFT/MCIntegration.jl)

MCIntegration.jl provides several Monte Carlo algorithms to calculate regular/singular integrals in finite or inifinite dimensions.  

# Quick start
The following examples demonstrate the basic usage of this package. 

## Convention
In general, high-dimensional integration may involve multiple integrals with multi-dimensional variables,
This package handles generic multiple integrals with multi-dimensional variables,
$$ f_i = \int d^{n_x}\vec{x} \int d^{n_y}\vec{y}... f(\vec{x}, \vec{y}...), \quad \text{with} i = 1, 2, ..., N$$
which may involve two types of variables: i) variables that is (almost) permutational symmetric in the integrand. They are called the symmetric variables, and are organized with vectors. This package allows nearly infinitely many 

In this package, we will can the variable vector as a pool of variables. ii) variables that are very different from each other, so that must be represented with different vectors such as $\vec{x}$ and $\vec{y}$.

The degree of freedom of the integrand is `dof` defined as [[$n_x$, $n_y$, ...], ....] with $N$ elements.

## Example 1. One-dimensional integral
We first demonstrate an example of highly singular integral. The following command evaluates $\int_0^1 \frac{\log (x)}{\sqrt{x}} dx = 4$.
```julia
julia> res = integrate((x, c)->log(x[1])/sqrt(x[1]), solver=:vegas, verbose=0) 
Integral 1 = -3.997980772652019 ± 0.0013607691354676158   (reduced chi2 = 1.93)

julia> report(res) #print out the iteration history
====================================     Integral 1    ==========================================
  iter              integral                            wgt average                  reduced chi2
-------------------------------------------------------------------------------------------------
ignore        -3.8394711 ± 0.12101621              -3.8394711 ± 0.12101621                 0.0000
     2         -3.889894 ± 0.04161423              -3.8394711 ± 0.12101621                 0.0000
     3        -4.0258398 ± 0.016628525              -4.007122 ± 0.015441393                9.2027
     4        -4.0010193 ± 0.0097242712            -4.0027523 ± 0.0082285382               4.6573
     5         -3.990754 ± 0.0055248673            -3.9944823 ± 0.0045868638               3.5933
     6         -4.000744 ± 0.0025751679            -3.9992433 ± 0.0022454867               3.0492
     7        -4.0021542 ± 0.005940518             -3.9996072 ± 0.0021004392               2.4814
     8        -3.9979708 ± 0.0034603885            -3.9991666 ± 0.0017955468               2.0951
     9         -3.994137 ± 0.0026675679            -3.9975984 ± 0.0014895459               2.1453
    10        -3.9999099 ± 0.0033455927            -3.9979808 ± 0.0013607691               1.9269
-------------------------------------------------------------------------------------------------
```
- By default, the function performs 10 iterations and each iteraction costs about `1e4` evaluations. You can adjust these values using `niter` and `neval` keywords arguments.

- The final result is obtained through an inverse-variance-weighted average of all iterations, excluding the first one (since there is no importance sampling yet!). The results are stored in the `res`, which is a [`Result`](https://numericaleft.github.io/MCIntegration.jl/dev/lib/montecarlo/#Main-module) struct, and you can access the statistics with `res.mean`, `res.stdev`, `res.chi2`, and `res.iterations`.

-  If you want to exclude more iterations from the final estimations, such as the first three iterations, you can call `Result(res, 3)` to get a new averaged result.

- After each iteration, the program adjusts a distribution to mimic the integrand, improving importance sampling. Consequently, the estimated integral from each iteration generally becomes more accurate with more iterations. As long as `neval` is sufficiently large, the estimated integrals from different iterations should be statistically independent, justifying an average of different iterations weighted by the inverse variance. The assumption of statistical independence can be explicitly verified with a chi-square test, in which the `chi2` (reduced $\chi^2$) value should be approximately one.

- The integrate function lets you choose a specific Monte Carlo (MC) algorithm by using the `solver` keyword argument. The example given employs the Vegas algorithm with `:vegas`. Additionally, this package provides two Markov-chain Monte Carlo (MCMC) algorithms for numerical integration: `:vegasmc` and `:mcmc`. Comparing these MCMC algorithms, `:vegasmc` offers better accuracy than `:mcmc` while keeping the same robustness. Although `:vegas` is generally slightly more accurate than `:vegasmc`, it is less robust. Considering the trade-off between accuracy and robustness, integrate defaults to using `:vegasmc`. For further information, consult the [Algorithm](#Algorithm) section.

- When defining your own integrand evaluation function, you need to provide two arguments: `(x, c)`:
  * `x` represents the integration variable, which by default falls within the range [0, 1). It should be considered as a pool of infinitely many random variables that follows the same distribution. To access the i-th random variable, use x[i]. For a better understanding, refer to Example 2 and the [Variables](#Variables) section.
  * `c` is a struct that holds the Monte Carlo (MC) configuration. This contains additional information that might be necessary for evaluating the integrand. For a practical example, see Example 5.

- For complex-valued integral, say with the type `ComplexF64`, you need to call `integrate(..., dtype = ComplexF64)` to specify the integrand data type. The error  of the real part and the imaginary part will be estimated independently.   

- You can suppress the output information by setting `verbose=-1`. If you want to see more information after the calculation, simply call `report(res)`. If you want to check the MC configuration, call `report(res.config)`.

## Example 2. Multi-dimensional integral: Symmetric Variables

In `MCIntegration.jl`, a variable is represented as a pool of random numbers drawn from the same distribution. For instance, you can explicitly initialize a set of variables in the range [0, 1) as follows:
```julia
julia> x=Continuous(0.0, 1.0) #Create a pool of continuous variables. 
Adaptive continuous variable in the domain [0.0, 1.0). Learning rate = 2.0.
```
This approach simplifies the evaluation of high-dimensional integrals involving multiple symmetric variables. For example, to calculate the area of a quarter unit circle (π/4 = 0.785398...):
```julia
julia> res = integrate((x, c)->(x[1]^2+x[2]^2<1.0); var = x, dof = [2, ]) 
Integral 1 = 0.7860119307731648 ± 0.002323473435947719   (reduced chi2 = 2.14)
```
If the integrand involve more than one variables, it is important to specify the `dof` vector. Each element of the `dof` vector represents the degrees of freedom of the corresponding integrand.

## Example 3. Multi-dimensional integral: Generic Variables
If the variables in a multi-dimensional integrand are not symmetric, it is better to define them as different types so that they can be sampled with different adaptive distributions. In the following example, we create a direct product of two continuous variables, then calculate a two-variable integral, 
```julia
julia> xy = Continuous([(0.0, 1.0), (0.0, 1.0)])
Adaptive CompositeVar{Tuple{Continuous{Vector{Float64}}, Continuous{Vector{Float64}}}} with 2 components.

julia> res = integrate(((x, y), c)-> log(x[1])/sqrt(x[1])*y[1]; var = xy)
Integral 1 = -2.0012850872834154 ± 0.001203058956026235   (reduced chi2 = 0.215)
```
The packed variable `xy` is of a type `CompositeVar` (see the [Variables](#Variables) section.). It is unpacked into a tuple of `x` and `y` within the integrand function. 

## Example 4. Evaluate Multiple Integrands Simultaneously
You can calculate multiple integrals simultaneously. If the integrands are similar to each other, evaluating the integrals simultaneously sigificantly reduces cost. The following example calculate the area of a quarter circle and the volume of one-eighth sphere.
```julia
julia> integrate((X, c)->(X[1]^2+X[2]^2<1.0, X[1]^2+X[2]^2+X[3]^2<1.0); var = Continuous(0.0, 1.0), dof = [[2,],[3,]])
Integral 1 = 0.7823432452235586 ± 0.003174967010742156   (reduced chi2 = 2.82)
Integral 2 = 0.5185515421806122 ± 0.003219487569949905   (reduced chi2 = 1.41)
```
Here `dof` defines how many (degrees of freedom) variables of each type. For example, [[n1, n2], [m1, m2], ...] means the first integral involves n1 varibales of type 1, and n2 variables of type2, while the second integral involves m1 variables of type 1 and m2 variables of type 2. The `dof` of the integrals can be quite different, the program will figure out how to optimally padding the integrands to match the degrees of freedom. 

You can also use the julia do-syntax to improve the readability of the above example,
```julia
julia> integrate(var = Continuous(0.0, 1.0), dof = [[2,], [3,]]) do X, c
           r1 = (X[1]^2 + X[2]^2 < 1.0) ? 1.0 : 0.0
           r2 = (X[1]^2 + X[2]^2 + X[3]^2 < 1.0) ? 1.0 : 0.0
           return (r1, r2)
       end
```

If there are too many components of integrands, it is better to preallocate the integrand weights. The function `integrate` provide an `inplace` key argument to achieve this goal. It is turned off by default, and only applies to the solver `:vegas` and `:vegasmc`. Once `inplace` is turned on, `integrate` will call the user-defined integrand function with a preallocated vector to store the user calculated weights. The following example demonstrates its usage,  
```julia
julia> integrate(var = Continuous(0.0, 1.0), dof = [[2,], [3,]], inplace=true) do X, f, c
           f[1] = (X[1]^2 + X[2]^2 < 1.0) ? 1.0 : 0.0
           f[2] = (X[1]^2 + X[2]^2 + X[3]^2 < 1.0) ? 1.0 : 0.0
       end
```

## Example 5. Use `Configuration` to Interface with MCIntegration 

- `Configuration` in integrands: As explained in the Example 1, the user-defined integrand has the signature `(x, c)` where `x` is the variable(s), and `c` is a ['Configuration'](https://numericaleft.github.io/MCIntegration.jl/dev/lib/montecarlo/#Main-module) struct stores the essential state information for the Monte Carlo sampling.Three particularly relavent members of `Configuratoin` include
  * `userdata` : if you pass a keyword argument `userdata` to the `integrate` function, then it will be stored here, so that you can access it in your integrand evaluation function. 
  * `var` : A tuple of variable(s). If there is only one variable in the tuple, then the first argument of the integrand will be `x = var[1]`. On the other hand, if there are multiple variables in the tuple, then `x = var`.
  * `obs` : A vector of observables. Each element is an accumulated estimator for one integrand. In other words, `length(obs)` = `length(dof)` = number of integrands.
  * `normalization`: the estimation of integrals are given by `obs ./ normalization`.

- `Configuration` in returned `Result`: The result returned by the `integrate` function contains the configuration after integration. If you want a detailed report, call `report(res.config)`. This configuration stores the optimized random variable distributions for the important sampling, which could be useful to evaluate other integrals with similar integrands. To use the optimized distributions, you can either call `integrate(..., config = res.config, ...)` to pass the entire configuration, or call `integrate(..., var = (res.config.var[1], ...), ...)` to pass one or more selected variables. In the following example, the second call is initialized with an optimized distribution, so that the first iteration is very accurate compared to the same row in the Example 1 output.
```julia
julia> res0 = integrate((x, c)->log(x[1])/sqrt(x[1]))
Integral 1 = -3.999299273090788 ± 0.001430447199375744   (chi2/dof = 1.46)

julia> res = integrate((x, c)->log(x[1])/sqrt(x[1]), verbose=0, config = res0.config)
====================================     Integral 1    ================================================
  iter              integral                            wgt average                      reduced chi2
-------------------------------------------------------------------------------------------------------
ignore        -4.0022708 ± 0.0044299263            -4.0022708 ± 0.0044299263               0.0000
     2        -3.9931774 ± 0.0042087902            -4.0022708 ± 0.0044299263               0.0000
     3        -4.0003596 ± 0.0026421611            -3.9983293 ± 0.0022377558               2.0889
     4        -3.9949943 ± 0.0027683518            -3.9970113 ± 0.0017402955               1.4833
     5        -4.0028234 ± 0.0035948238            -3.9981148 ± 0.0015663954               1.6948
     6        -4.0037708 ± 0.0021567542             -4.000068 ± 0.0012674021               2.3967
     7        -3.9946345 ± 0.0040640646            -3.9995864 ± 0.0012099316               2.2431
     8        -4.0039064 ± 0.0032909285            -4.0001008 ± 0.0011356123               2.1223
     9        -3.9959395 ± 0.0036121885            -3.9997265 ± 0.0010833368               1.9916
    10        -3.9955869 ± 0.0032874678             -3.999321 ± 0.0010289098               1.9215
-------------------------------------------------------------------------------------------------------
Integral 1 = -3.9993209996786128 ± 0.0010289098118216647   (reduced chi2 = 1.92)
```

## Example 6. Measure Histogram
You may want to study how an integral changes with a tuning parameter. The following example is how to solve the histogram measurement problem.
```julia
julia> N = 20;

julia> grid = [i / N for i in 1:N];

julia> function integrand(vars, config)
            grid = config.userdata # radius
            x, bin = vars #unpack the variables
            r = grid[bin[1]] # binned variable in [0, 1)
            r1 = x[1]^2 + r^2 < 1 # circle
            r2 = x[1]^2 + x[2]^2 + r^2 < 1 # sphere
            return r1, r2
        end;

julia> function measure(vars, obs, weights, config) 
       # obs: prototype of the observables for each integral
           x, bin = vars #unpack the variables
           obs[1][bin[1]] += weights[1] # circle
           obs[2][bin[1]] += weights[2] # sphere
       end;

julia> res = integrate(integrand;
                measure = measure, # measurement function
                var = (Continuous(0.0, 1.0), Discrete(1, N)), # a continuous and a discrete variable pool 
                dof = [[1,1], [2,1]], 
                # integral-1: one continuous and one discrete variables, integral-2: two continous and one discrete variables
                obs = [zeros(N), zeros(N)], #  observable prototypes of each integral
                userdata = grid, neval = 1e5)
Integral 1 = 0.9957805541613277 ± 0.008336657854575344   (reduced chi2 = 1.15)
Integral 2 = 0.7768105610812656 ± 0.006119386106596811   (reduced chi2 = 1.4)
```
You can visualize the returned result `res` with `Plots.jl`. The commands `res.mean[i]` and `res.stdev[i]` give the mean and stdev of the histogram of the `i`-th integral.
```julia
julia> using Plots

julia> plt = plot(grid, res.mean[1], yerror = res.stdev[1], xlabel="R", label="circle", aspect_ratio=1.0, xlim=[0.0, 1.0])

julia> plot!(plt, grid, res.mean[2], yerror = res.stdev[2], label="sphere")
```
![histogram](docs/src/assets/circle_sphere.png?raw=true "Circle and Sphere")

# Algorithm

This package provides three solvers.

- Vegas algorithm (`:vegas`): A Monte Carlo algorithm that uses importance sampling as a variance-reduction technique. Vegas iteratively builds up a piecewise constant weight function, represented
on a rectangular grid. Each iteration consists of a sampling step followed by a refinement
of the grid. The exact details of the algorithm can be found in **_G.P. Lepage, J. Comp. Phys. 27 (1978) 192, 3_** and
**_G.P. Lepage, Report CLNS-80/447, Cornell Univ., Ithaca, N.Y., 1980_**. 

- Vegas algorithm based on Markov-chain Monte Carlo (`:vegasmc`): A markov-chain Monte Carlo algorithm that uses the Vegas variance-reduction technique. It is as accurate as the vanilla Vegas algorithm, meanwhile tends to be more robust. For complicated high-dimensional integral, the vanilla Vegas algorithm can fail to learn the piecewise constant weight function. This algorithm uses Metropolis–Hastings algorithm to sample the integrand and improves the weight function learning.

- Markov-chain Monte Carlo (`:mcmc`): This algorithm is useful for calculating bundled integrands that are too many to calculate at once. Examples are the path-integral of world lines of quantum particles, which involves hundreds and thousands of nested spacetime integrals. This algorithm uses the Metropolis-Hastings algorithm to jump between different integrals so that you only need to evaluate one integrand at each Monte Carlo step. Just as `:vegas` and `:vegasmc`, this algorithm also learns a piecewise constant weight function to reduce the variance. However, because it assumes you can access one integrand at each step, it tends to be less accurate than the other two algorithms for low-dimensional integrals.   

The signature of the integrand and measure functions of the `:mcmc` solver receices an additional index argument than that of the `:vegas` and `:vegasmc` solvers. As shown in the above examples, the integrand and measure functions of the latter two solvers should be like `integrand(vars, config)` and `measure(vars, obs, weights, config)`, where `weights` is a vectors carries the values of the integrands at the current MC step. On the other hand, the `:mcmc` solver requires something like `integrand(idx, vars, config)` and `measure(idx, vars, weight, config)`, where `idx` is the index of the integrand of the current step, and the argument `weight` is a scalar carries the value of the current integrand being sampled.

# Variables

The package supports a couple of common types random variables. You can create them using the following constructors,

- `Continous(lower, upper[; adapt = true, alpha = 3.0, ...])`: Continuous real-valued variables on the domain [lower, upper). MC will learn the distribution using the Vegas algorithm and then perform an imporant sampling accordingly.
- `Discrete(lower::Int, upper::Int[; adapt = true, alpha = 3.0, ...])`: Integer variables in the closed set [lower, upper]. MC will learn the distribution and perform an imporant sampling accordingly.

After each iteration, the code will try to optimize how the variables are sampled, so that the most important regimes of the integrals will be sampled most frequently. Setting `alpha` to be true/false will turn on/off this distribution learning. The parameter `alpha` controls the learning rate.

When you call the above constructor, it creates an unlimited pool of random variables of a given type. The size of the pool will be dynamically determined when you call a solver. All variables in this pool will be sampled with the same distribution. In many high-dimensional integrals, many integration variables may contribute to the integral in a similar way; then they can be sampled from the same variable pool. For example, in the above code example, the integral for the circle area and the sphere volume both involve the variable type `Continuous`. The former has dof=2, while the latter has dof=3. To evaluate a given integrand, you only need to choose some of the variables to evaluate a given integral. The rest of the variables in the pool serve as dummy ones. They will not cause any computational overhead.

The variable pool trick will significantly reduce the cost of learning their distribution. It also opens the possibility of calculating integrals with infinite dimensions (for example, the path-integral of particle worldlines in quantum many-body physics). 

If some of the variables are paired with each other (for example, the three continuous variables (r, θ, ϕ) representing a 3D vector), then you can pack them into a joint random variable, which can be constructed with the following constructor,
- `CompositeVar(var1, var2, ...[; adapt = true, alpha = 3.0, ...])`: A product of different types of random variables. It samples `var1`, `var2`, ... with their producted distribution. 

If the packed variables are all continuous of discrete, then you can create them in a more straightforward way,
- `Continous([(lower1, upper1), (lower2, upper2), ...], [; adapt = true, alpha = 3.0, ...])`.
- `Discrete([(lower1, upper1), (lower2, upper2), ...], [; adapt = true, alpha = 3.0, ...])`.

The packed variables will be sampled all together in the Markov-chain based solvers (`:vegasmc` and `:mcmc`). Such updates will generate more independent samples compared to the unpacked version. Sometimes, it could reduce the auto-correlation time of the Markov chain and make the algorithm more efficient.

Moreover, packed variables usually indicate nontrivial correlations between their distributions. In the future, it will be interesting to learn such correlation so that one can sample the packed variables more efficiently.

# Parallelization
MCIntegration supports both MPI and multi-thread parallelization. You can even mix them if necessary.

## MPI
To run your code in MPI mode, simply use the command,
```bash
mpiexec -n #NCPU julia ./your_script.jl
```
where `#NCPU` is the number of workers. Internally, the MC sampler will send the blocks (controlled by the argument `Nblock`, see above example code) to different workers, then collect the estimates in the root node. 

Note that you need to install the package [MPI.jl](https://github.com/JuliaParallel/MPI.jl) to use the MPI mode. See this [link](https://juliaparallel.github.io/MPI.jl/stable/configuration/) for the instruction on the configuration.

The user essentially doesn't need to write additional code to support the parallelization. The only tricky part is the output: only the function `MCIntegratoin.integrate` of the root node returns meaningful estimates, while other workers simply returns `nothing`.

## Multi-threading

MCIntegration supports multi-threading with or without MPI. To run your code with multiple threads, start Julia with
```bash
julia -t #NCPU ./your_script.jl
```
Note that all threads will share the same memory. The user-defined `integrand` and `measure` functions should be implemented thread-safe (for example, be very careful about reading any data if another thread might write to it). We recommend the user read Julia's official [documentation](https://docs.julialang.org/en/v1/manual/multi-threading/).

There are two different ways to parallelize your code with multiple threads. 

1. If you need to evaluate multiple integrals, each thread can call the function `MCIntegration.integrate` to do one integral. In the following example, we use three threads to evaluate three integrals altogether. Note that only three threads will be used even if you initialize Julia with more than three threads.
```julia
julia> Threads.@threads for i = 1:3
       println("Thread $(Threads.threadid()) returns ", integrate((x, c) -> x[1]^i, verbose=-2))
       end
Thread 2 returns Integral 1 = 0.24995156136254149 ± 6.945088534643841e-5   (reduced chi2 = 2.95)
Thread 3 returns Integral 1 = 0.3334287563137184 ± 9.452648803649706e-5   (reduced chi2 = 1.35)
Thread 1 returns Integral 1 = 0.5000251243601586 ± 0.00013482206569391864   (reduced chi2 = 1.58)
```

2. Only the main thread calls the function `MCIntegration.integrate`, then parallelize the internal blocks with multiple threads. To do that, you need to call the function `MCIntegration.integrate` with a key argument `parallel = :thread`. This approach will utilize all Julia threads.  For example,
```julia
julia> for i = 1:3
       println("Thread $(Threads.threadid()) return ", integrate((x, c) -> x[1]^i, verbose=-2, parallel=:thread))
       end
Thread 1 return Integral 1 = 0.5001880440214347 ± 0.00015058935731086765   (reduced chi2 = 0.397)
Thread 1 return Integral 1 = 0.33341068551139696 ± 0.00010109649819894601   (reduced chi2 = 1.94)
Thread 1 return Integral 1 = 0.24983868976137244 ± 8.546009018501706e-5   (reduced chi2 = 1.54)
```
