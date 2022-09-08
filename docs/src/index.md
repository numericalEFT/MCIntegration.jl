```@meta
CurrentModule = MCIntegration
```

# MCIntegration

Robust and efficient Monte Carlo calculator for high-dimensional integral.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://numericalEFT.github.io/MCIntegration.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://numericalEFT.github.io/MCIntegration.jl/dev)
[![Build Status](https://github.com/numericalEFT/MCIntegration.jl/workflows/CI/badge.svg)](https://github.com/numericalEFT/MCIntegration.jl/actions)
[![Coverage](https://codecov.io/gh/numericalEFT/MCIntegration.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/numericalEFT/MCIntegration.jl)

MCIntegration.jl provides several Monte Carlo algorithms to calculate regular/singular integrals in finite or inifinite dimensions.  

# Quick start
The following examples demonstrate the basic usage of this package. 

## One-dimensional integral
We first show an example of highly singular integral. The following command evaluates ∫_0^1 log(x)/√x dx = 4.
```julia
julia> integrate((x, c)->log(x[1])/sqrt(x[1]), solver=:vegas) 
==================================     Integral 1    ==============================================
  iter          integral                            wgt average                          chi2/dof
---------------------------------------------------------------------------------------------------
     1       -4.3455199 ± 0.27334819              -4.3455199 ± 0.27334819                  0.0000
     2       -3.8784897 ± 0.030889492             -3.8843784 ± 0.030694133                 2.8824
     3       -3.9967001 ± 0.016241895             -3.9721295 ± 0.014355926                 6.6721
     4       -4.0025717 ± 0.012134258             -3.9898859 ± 0.0092672832                5.3223
     5       -4.0090158 ± 0.0042303036            -4.0057171 ± 0.0038483213                4.8733
     6       -4.0000401 ± 0.0043113376            -4.0031997 ± 0.0028709675                4.0916
     7       -4.0043599 ± 0.0034347516            -4.0036769 ± 0.0022027999                3.4209
     8       -4.0023797 ± 0.0026638972            -4.0031501 ± 0.0016975893                2.9523
     9       -3.9995249 ± 0.0027579019             -4.002154 ± 0.001445668                 2.7399
    10       -4.0009821 ± 0.0020086867            -4.0017541 ± 0.0011733717                2.4604
---------------------------------------------------------------------------------------------------
Integral 1 = -4.0017540835419 ± 0.0011733716863794182   (chi2/dof = 2.46)
```
By default, the function performs 10 iterations and each iteraction costs about `1e4` evaluations. You may reset these values with `niter` and `neval` keywords arguments.

Internally, the `integrate` function optimizes the important sampling after each iteration. The results generally improves with iteractions. As long as `neval` is sufficiently large, the estimations from different iteractions should be statistically independent. This will justify an average of different iterations weighted by the inverse variance. The assumption of statically independence can be explicitly checked with chi-square test, namely `chi2/dof` should be about one. 

You can pass the keyword arguemnt `solver` to the `integrate` functoin to specify the Monte Carlo algorithm. The above examples uses the Vegas algorithm with `:vegas`. In addition, this package provides two Markov-chain Monte Carlo algorithms for numerical integration. You can call them with `:vegasmc` or `:mcmc`. Check the Algorithm section for more details. 

For `:vegas` and `vegasmc` algorithm, the user-defined integrand function should have two arguments `(x, c)`, where `x` is the integration variable, while `c` is a struct of the MC configuration. It contains additional information which may be needed for integrand evalution. For example, if you pass a keyword argument `userdata` to the `integrate` function, then you can access the userdata within your integrand function using `c.userdata`. 

## Multi-dimensional integral
The following example first defines a pool of variables in [0, 1), then evaluates the area of a quarter unit circle (π/4 = 0.785398...).
```julia
julia> X=Continuous(0.0, 1.0) #Create a pool of continuous variables. It supports as much as 16 same type of variables. see the section [variable](#variable) for more details.
Adaptive continuous variable in the domain [0.0, 1.0). Max variable number = 16. Learning rate = 2.0.

julia> integrate((X, c)->(X[1]^2+X[2]^2<1.0); var = X, dof = 2, print=-1) # print=-1 minimizes the output information
Integral 1 = 0.7832652785953883 ± 0.002149843816733503   (chi2/dof = 1.28)
```

## Multiple Multi-dimensional integrals
You can calculate multiple integrals simultaneously. If the integrands are similar to each other, evaluating the integrals simultaneously sigificantly reduces cost. The following example calculate the area of a quarter circle and the volume of one-eighth sphere.
```julia
julia> integrate((X, c)->(X[1]^2+X[2]^2<1.0, X[1]^2+X[2]^2+X[3]^2<1.0); var = Continuous(0.0, 1.0), dof = [[2,],[3,]], print=-1)
Integral 1 = 0.7823432452235586 ± 0.003174967010742156   (chi2/dof = 2.82)
Integral 2 = 0.5185515421806122 ± 0.003219487569949905   (chi2/dof = 1.41)
```
Here `dof` defines how many (degrees of freedom) variables of each type. For example, [[n1, n2], [m1, m2], ...] means the first integral involves n1 varibales of type 1, and n2 variables of type2, while the second integral involves m1 variables of type 1 and m2 variables of type 2. The `dof` of the integrals can be quite different, the program will figure out how to optimally padding the integrands to match the degrees of freedom. 

You can also use the julia do-syntax to improve the readability of the above example,
```julia
julia> integrate(var = Continuous(0.0, 1.0), dof = [[2,], [3,]], print = -1) do X, c
           r1 = (X[1]^2 + X[2]^2 < 1.0) ? 1.0 : 0.0
           r2 = (X[1]^2 + X[2]^2 + X[3]^2 < 1.0) ? 1.0 : 0.0
           return (r1, r2)
       end
```

## Histogram measurement
You may want to study how an integral changes with a tuning parameter. The following example how to solve histogram measurement problem.
```julia
julia> N = 20;

julia> grid = grid = [i / N for i in 1:N];

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
                obs = [zeros(N), zeros(N)], # prototype of the observables for each integral
                userdata = grid, neval = 1e5, print = -1)
Integral 1 = 0.9957805541613277 ± 0.008336657854575344   (chi2/dof = 1.15)
Integral 2 = 0.7768105610812656 ± 0.006119386106596811   (chi2/dof = 1.4)
```
You can visualize the returned result `res` with `Plots.jl`. The command `res.mean[i]` and `res.stdev[i]` give the mean and stdev of the histogram of the integral `i`.
```julia
julia> using Plots

julia> plt = plot(grid, res.mean[1], yerror = res.stdev[1], xlabel="R", label="circle", aspect_ratio=1.0, xlim=[0.0, 1.0])

julia> plot!(plt, grid, res.mean[2], yerror = res.stdev[2], label="sphere")
```
![histogram](assets/circle_sphere.png?raw=true "Circle and Sphere")

# Algorithm

This package provides three solvers.

- Vegas algorithm (`:vegas`): A Monte Carlo algorithm that uses importance sampling as a variance-reduction technique. Vegas iteratively builds up a piecewise constant weight function, represented
on a rectangular grid. Each iteration consists of a sampling step followed by a refinement
of the grid. The exact details of the algorithm can be found in **_G.P. Lepage, J. Comp. Phys. 27 (1978) 192, 3_** and
**_G.P. Lepage, Report CLNS-80/447, Cornell Univ., Ithaca, N.Y., 1980_**. 

- Vegas algorithm based on Markov-chain Monte Carlo (`:vegasmc`): A markov-chain Monte Carlo algorithm that uses the Vegas variance-reduction technique. It is as accurate as the vanilla Vegas algorithm, meanwhile tends to be more robust. For complicated high-dimensional integral, the vanilla Vegas algorithm can fail to learn the piecewise constant weight function. This algorithm uses Metropolis–Hastings algorithm to sample the integrand and improves the weight function learning.

- Markov-chain Monte Carlo (`:mcmc`): This algorithm is useful for calculating bundled integrands that are too many to calculate at once. Examples are the path-integral of world lines of quantum particles, which involves hundreds and thousands of nested spacetime integrals. This algorithm uses the Metropolis-Hastings algorithm to jump between different integrals so that you only need to evaluate one integrand at each Monte Carlo step. Just as `:vegas` and `:vegasmc`, this algorithm also learns a piecewise constant weight function to reduce the variance. However, because it assumes you can access one integrand at each step, it tends to be less accurate than the other two algorithms for low-dimensional integrals.   

The signature of the integrand and measure functions of the `:mcmc` solver receices an additional index argument than that of the `:vegas` and `:vegasmc` solvers. As shown in the above examples, the integrand and measure functions of the latter two solvers should be like `integrand( vars, config)` and `measure(vars, obs, weights, config)`, where `weights` is a vectors carries the values of the integrands at the current MC step. On the other hand, the `:mcmc` solver requires something like `integrand(idx, vars, config)` and `measure(idx, vars, weight, config)`, where `idx` is the index of the integrand of the current step, and the argument `weight` is a scalar carries the value of the current integrand being sampled.

# Variables

The package supports a couple of common types random variables. You can create them using the following constructors,

- `Continous(lower, upper[; adapt = true, alpha = 3.0, ...])`: Continuous real-valued variables on the domain [lower, upper). MC will learn the distribution using the Vegas algorithm and then perform an imporant sampling accordingly.
- `Discrete(lower::Int, upper::Int[; adapt = true, alpha = 3.0, ...])`: Integer variables in the closed set [lower, upper]. MC will learn the distribution and perform an imporant sampling accordingly.

After each iteration, the code will try to optimize how the variables are sampled, so that the most important regimes of the integrals will be sampled most frequently. Setting `alpha` to be true/false will turn on/off this distribution learning. The parameter `alpha` controls the learning rate.

When you call the above constructor, it creates a unlimited pool of random variables of a given type. The size of the pool will be dynamically determined when you call a solver. All variables in this pool will be sampled with the same distribution. In many high-dimensional integrals, many variables of integration may contribute to the integral in a similar way, then they can be sampled from the same variable pool. For example, in the above code example, the integral for the circle area and the sphere volume both involve the variable type `Continuous`. The former has dof=2, while the latter has dof=3. To evaluate a given integrand, you only need to choose some of variables to evaluate a given integral. The rest of the variables in the pool serve as dummy ones. They will not cause any computational overhead.

The variable pool trick will siginicantly reduce the cost of learning their distribution. It also opens the possibility to calculate integrals with infinite dimensions (for example, the path-integral of particle worldlines in quantum many-body physics). 

If some of the variables are paired with each other (for example, the three continuous variables (r, θ, ϕ) representing a 3D vector), then you can pack them into a joint random variable, which can be constructed with the following constructor,
- `CompositeVar(var1, var2, ...[; adapt = true, alpha = 3.0, ...])`: A produce of different types of random variables. It samples `var1`, `var2`, ... with their producted distribution. 

The packed variables will be sampled all together in the Markov-chain based solvers (`:vegasmc` and `:mcmc`). Such updates will generate more independent samples compared to the unpacked version. Sometimes, it could reduce the auto-correlation time of the Markov chain and make the algorithm more efficient.

Moreover, packed variables usually indicate nontrivial correlations between their distributions. In the future, it will be interesting to learn such correlation so that one can sample the packed variables more efficiently.

!!! tip "Integrate over different domains"

    If you want to compute an integral over a semi-infinite or an inifite domain, you
    can follow [this advice](http://ab-initio.mit.edu/wiki/index.php/Cubature#Infinite_intervals)
    from Steven G. Johnson: to compute an integral over a semi-infinite interval,
    you can perform the change of variables ``x=a+y/(1-y)``:

    ```math
    \int_{a}^{\infty} f(x)\,\mathrm{d}x = \int_{0}^{1}
    f\left(a + \frac{y}{1 - y}\right)\frac{1}{(1 - y)^2}\,\mathrm{d}y
    ```

    For an infinite interval, you can perform the change of variables ``x=(2y -
    1)/((1 - y)y)``:

    ```math
    \int_{-\infty}^{\infty} f(x)\,\mathrm{d}x = \int_{0}^{1}
    f\left(\frac{2y - 1}{(1 - y)y}\right)\frac{2y^2 - 2y + 1}{(1 -
    y)^2y^2}\,\mathrm{d}y
    ```

    In addition, recall that for an [even function](https://en.wikipedia.org/wiki/Even_and_odd_functions#Even_functions)
    ``\int_{-\infty}^{\infty} f(x)\,\mathrm{d}x = 2\int_{0}^{\infty}f(x)\,\mathrm{d}x``, 
    while the integral of an [odd function](https://en.wikipedia.org/wiki/Even_and_odd_functions#Odd_functions)
    over the infinite interval ``(-\infty, \infty)`` is zero.

# Parallelization

MCIntegration supports MPI parallelization. To run your code in MPI mode, simply use the command
```bash
mpiexec julia -n #NCPU ./your_script.jl
```
where `#CPU` is the number of workers. Internally, the MC sampler will send the blocks (controlled by the argument `Nblock`, see above example code) to different workers, then collect the estimates in the root node. 

Note that you need to install the package [MPI.jl](https://github.com/JuliaParallel/MPI.jl) to use the MPI mode. See this [link](https://juliaparallel.github.io/MPI.jl/stable/configuration/) for the instruction on the configuration.

The user essentially doesn't need to write additional code to support the parallelization. The only tricky part is the output: only the function `MCIntegratoin.integrate` of the root node returns meaningful estimates, while other workers simply returns `nothing`. 
