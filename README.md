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
We first show an example of highly singular integral. The following command evaluates $\int_0^1 \frac{\operatorname{log}(x)}{\sqrt{x}} dx = 4$.
```julia
julia> res = integrate((x, c)->log(x[1])/sqrt(x[1]), solver=:vegas) 
Integral 1 = -3.997980772652019 ± 0.0013607691354676158   (chi2/dof = 1.93)

julia> report(res) #print out the iteration history
====================================     Integral 1    ==========================================
  iter              integral                            wgt average                      chi2/dof
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
- By default, the function performs 10 iterations and each iteraction costs about `1e4` evaluations. You may reset these values with `niter` and `neval` keywords arguments.

- The final result is obtained by inverse-variance-weighted averge of all the iterations except the first one (there is no important sampling yet!). They are stored in the return value`res`, which is a struct [`Result`](https://numericaleft.github.io/MCIntegration.jl/dev/lib/montecarlo/#Main-module). You can access the statistics with `res.mean`, `res.stdev`, `res.chi2`, `res.dof` and `res.iterations` for all the iterations. 

-  If you want to exclude more iterations, say the first three iterations, you can get a new result with the call `Result(res, 3)`.  

- Internally, the `integrate` function optimizes the important sampling after each iteration. The results generally improves with iteractions. As long as `neval` is sufficiently large, the estimations from different iteractions should be statistically independent. This will justify an average of different iterations weighted by the inverse variance. The assumption of statically independence can be explicitly checked with chi-square test, namely `chi2/dof` should be about one. 

- You can pass the keyword arguemnt `solver` to the `integrate` functoin to specify the Monte Carlo algorithm. The above examples uses the Vegas algorithm with `:vegas`. In addition, this package provides two Markov-chain Monte Carlo algorithms for numerical integration. You can call them with `:vegasmc` or `:mcmc`. Check the Algorithm section for more details. 

- For the `:vegas` and `vegasmc` algorithms, the user-defined integrand evaluation function requires two arguments `(x, c)`, where `x` is the integration variable, while `c` is a struct stores the MC configuration. The latter contains additional information which may be needed for integrand evalution.  

- The ['Configuration'](https://numericaleft.github.io/MCIntegration.jl/dev/lib/montecarlo/#Main-module) struct stores the essential state information for the Monte Carlo sampling. Two particularly relavent members are
  * `userdata` : if you pass a keyword argument `userdata` to the `integrate` function, then it will be stored here, so that you can access it in your integrand evaluation function. 
  * `var` : A tuple of variables. In the above example, `var = (x, )` so that `var[1] === x`. 

- The result returned by the `integrate` function contains the configuration after integration. If you want a detailed report, call `report(res.config)`. This configuration stores the optimized random variable distributions for the important sampling, which could be useful to evaluate other integrals with similar integrands. To use the optimized distributions, you can either call `integrate(..., config = res.config, ...)` to pass the entire configuration, or call `integrate(..., var = (res.config.var[1], ...), ...)` to pass one or more selected variables.

## Multi-dimensional integral
The following example first defines a pool of variables in [0, 1), then evaluates the area of a quarter unit circle (π/4 = 0.785398...).
```julia
julia> X=Continuous(0.0, 1.0) #Create a pool of continuous variables. 
Adaptive continuous variable in the domain [0.0, 1.0). Max variable number = 16. Learning rate = 2.0.

julia> res = integrate((X, c)->(X[1]^2+X[2]^2<1.0); var = X, dof = 2) 
Integral 1 = 0.7860119307731648 ± 0.002323473435947719   (chi2/dof = 2.14)
```
Here we suppress the output information by `print=-1`. If you want to see more information after the calculation, simply call `report(res)`. If you want to check the MC configuration, you may call `report(res.config)`.

## Multiple Integrands Simultaneously
You can calculate multiple integrals simultaneously. If the integrands are similar to each other, evaluating the integrals simultaneously sigificantly reduces cost. The following example calculate the area of a quarter circle and the volume of one-eighth sphere.
```julia
julia> integrate((X, c)->(X[1]^2+X[2]^2<1.0, X[1]^2+X[2]^2+X[3]^2<1.0); var = Continuous(0.0, 1.0), dof = [[2,],[3,]])
Integral 1 = 0.7823432452235586 ± 0.003174967010742156   (chi2/dof = 2.82)
Integral 2 = 0.5185515421806122 ± 0.003219487569949905   (chi2/dof = 1.41)
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

## Measure Histogram
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
Integral 1 = 0.9957805541613277 ± 0.008336657854575344   (chi2/dof = 1.15)
Integral 2 = 0.7768105610812656 ± 0.006119386106596811   (chi2/dof = 1.4)
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

The packed variables will be sampled all together in the Markov-chain based solvers (`:vegasmc` and `:mcmc`). Such updates will generate more independent samples compared to the unpacked version. Sometimes, it could reduce the auto-correlation time of the Markov chain and make the algorithm more efficient.

Moreover, packed variables usually indicate nontrivial correlations between their distributions. In the future, it will be interesting to learn such correlation so that one can sample the packed variables more efficiently.

# Parallelization

MCIntegration supports MPI parallelization. To run your code in MPI mode, simply use the command
```bash
mpiexec julia -n #NCPU ./your_script.jl
```
where `#NCPU` is the number of workers. Internally, the MC sampler will send the blocks (controlled by the argument `Nblock`, see above example code) to different workers, then collect the estimates in the root node. 

Note that you need to install the package [MPI.jl](https://github.com/JuliaParallel/MPI.jl) to use the MPI mode. See this [link](https://juliaparallel.github.io/MPI.jl/stable/configuration/) for the instruction on the configuration.

The user essentially doesn't need to write additional code to support the parallelization. The only tricky part is the output: only the function `MCIntegratoin.integrate` of the root node returns meaningful estimates, while other workers simply returns `nothing`.
