# MCIntegration.jl: Monte Carlo Integration in Julia

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://numericalEFT.github.io/MCIntegration.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://numericalEFT.github.io/MCIntegration.jl/dev)
[![Build Status](https://github.com/numericalEFT/MCIntegration.jl/workflows/CI/badge.svg)](https://github.com/numericalEFT/MCIntegration.jl/actions)
[![Coverage](https://codecov.io/gh/numericalEFT/MCIntegration.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/numericalEFT/MCIntegration.jl)

## Why Choose MCIntegration.jl?
MCIntegration.jl is a comprehensive Julia package designed to handle both regular and singular high-dimensional integrals with ease. Its implementation of robust Monte Carlo integration methods makes it a versatile tool in various scientific domains, including high-energy physics, material science, computational chemistry, financial mathematics, and machine learning.

The high-level simplicity and flexibility of Julia combined with the performance capabilities of C/C++-like compiled languages make it a fantastic choice for implementing Monte Carlo methods. Monte Carlo methods, which require extensive computations, can greatly benefit from Julia's just-in-time (JIT) compilation that allows MCIntegration.jl to perform calculations at a near-C/C++ efficiency. Moreover, the intuitive high-level syntax of Julia allows users to define their integrands effortlessly, adding to the customizability and user-friendliness of MCIntegration.jl.

## Features and Benefits
- **Monte Carlo Integration:** Estimate the value of complex integrals using Monte Carlo methods, a class of algorithms suitable for high-dimensional integrals.
- **Variable Handling:** The package offers unique handling of symmetric and asymmetric variables with an efficient 'variable pool' concept, enabling optimization of computations.
- **Selection of Algorithms:** Choose between three Monte Carlo integration solvers - `Vegas`, `VegasMC`, and `MCMC`, each tailored for different types of integral evaluations.
- **Parallelization:** Accelerate your computations using multi-threading and MPI capabilities for parallel computing.

## Installation
To install MCIntegration.jl, use Julia's package manager. Open the Julia REPL, type `]` to enter the package mode, and then:
```
pkg> add MCIntegration
```
## Quick Start
MCIntegration.jl simplifies complex integral calculations. Here are two examples to get you started.

To estimate the integral $\int_0^1 \frac{\log(x)}{\sqrt{x}} dx = 4$, you can use:
```julia
julia> f(x, _) = log(x[1]) / sqrt(x[1])   # Define your integrand where 'x' represents the random variable in the integral
julia> integrate(f, var = Continuous(0, 1), neval=1e5)   # Perform the MC integration for 1e5 steps where 'var' is used to specify the type and range of the random variable 'x'
Integral 1 = -3.99689518016736 ± 0.001364833686666744   (reduced chi2 = 0.695)
```
In this example, we define an integrand function `f(x, c)` where `x` represents the random variables in the integral and `c` is a `Configuration` object parameter that can hold extra parameters that might be necessary for more complex integrand functions. The variable `x` is determined by the var parameter in `integrate()`.

`MCIntegration.jl` also supports Discrete variables. For instance, let's estimate $\pi$ through the Taylor series for $\pi/4 = 1 - 1/3 + 1/5 -1/7 + 1/9 - ...$:
```julia

julia> term(n, c) = 4 * ((-1)^(n[1]+1)) / (2*n[1] - 1)  # Define your term function where 'n' represents the discrete variable in the integral
julia> integrate(term; var = Discrete(1, 100), neval = 1e5)  # Perform the MC integration for 1e5 steps where 'var' is used to specify the type and range of the discrete variable 'n'
Integral 1 = 3.120372107250909 ± 0.016964643375124093   (reduced chi2 = 1.38)
```

## Understanding Variables
To handle more complex integrals, it's necessary to understand how `MCIntegration.jl` designs and uses variables. `MCIntegration.jl` can handle multiple integrals with multi-dimensional variables in the general form
$$ \int d\vec{x} \int d\vec{y}... \vec{f}(\vec{x}, \vec{y}...)$$
where for discrete variables, the integrals should be regarded as summations.

In `MCIntegration.jl`, the "degree of freedom" (`dof`) defines the number of each type of variables that your integrand function needs. For each integrand, it is represented as a list of dimensions for each type of variables, like dof($\vec{f}$) = [[dim($\vec{x}$), dim($\vec{y}$), ...], ...], which contains dim($\vec{f}$) elements.

Variables can be categorized into three types:

- **Symmetric Variables:** These are assembled into vectors. Each vector serves as an **unlimited** pool of variables. All variables within the same pool are sampled with the same optimized distributions.

- **Asymmetric Variables:** These are distinct and are represented by different vectors, each with its own distribution.

Composite Variables: These are a set of variables that are consistently grouped together across all integrands, sharing the same dof. They are updated together during the integration process, leading to computational efficiency especially when working with MCMC-based solvers.

- **Symmetric Variables** are organized into vectors, and each such vector acts as an unlimited **pool** of variables. `MCIntegration.jl` samples all variables in the pool with the same optimized distributions. 

- **Asymmetric Variables** significantly differ from each other, so they must be represented by different vectors and sampled with different distributions.

- **Composite Variables** refer to a group of variables consistently used together across all integrands, sharing the same `dof`. Composite variables can be packed and updated together during the integration process, offering computational efficiency when working with MCMC-based solvers (see [`Selecting Algorithms`] section).

Here are examples to illustrate the usage of different types of variables:

The following example estimates $\pi$ with two symmetric variables `x[1]` and `x[2]` both range uniformly from 0 to 1.
```julia
julia> f(x, c) = x[1]^2 + x[2]^2 < 1
julia> integrate(f; var = Continuous(-1, 1), dof = [[2, ],]) # dof must be provided for multi-dimensional integrands
Integral 1 = 3.1316915341619413 ± 0.008785871829296759   (reduced chi2 = 0.298)
```
The same problem can be solved in the polar coordinates $(r, \theta)$, which are asymmetric variables that are better sampled with different distributions,
```julia
julia> g((r, θ), c) = r[1] # Upack the variables into r and θ. The integrand is independent of θ.
julia> integrate(g; var = (Continuous(0, 1), Continuous(0, 2π)), dof = [(1, 1),]) #asymmetric variables
Integral 1 = 3.1416564680126626 ± 0.0035638975370485427   (reduced chi2 = 1.94) 

# alternatively, you may create (r, θ) as a Composite Variable
julia> integrate(g; var = Continuous([(0, 1), (0, 2π)]) , dof = [(1, ),]) 
# equvilantly, use the constructor: CompositeVar(Continuous(0, 1), Continuous(0, 2π))
Integral 1 = 3.14367422926071 ± 0.0011572440016582415   (reduced chi2 = 0.735)
```

## Selecting Algorithms

MCIntegration.jl offers three Monte Carlo integration algorithms, all of which leverage the Vegas map technique for importance sampling. This approach constructs a piecewise constant Vegas map, a probability distribution function approximating the shape of the integrand to enhance the efficiency of the integral estimation.

Here's a brief overview of the three solvers:

1. **Vegas (`:vegas`):** The classic Vegas algorithm uses the Monte Carlo method and samples all integrands across all variables simultaneously at each step. It is efficient for low-dimensional integrals but might struggle with high-dimensional ones where the Vegas map fails to accurately mimic the integrand's shape.

2. **Vegas with MCMC (`:vegasmc`):** This innovative solver, first introduced in MCIntegration.jl, combines Vegas with Markov-chain Monte Carlo. This hybrid approach provides a robust solution, especially for intricate, high-dimensional integrals. In the Vegas MC approach, a variable is selected randomly, and a Metropolis-Hastings algorithm is utilized to propose a new variable based on the Vegas map. This update is applied simultaneously across all integrands, improving robustness when the Vegas map struggles with approximating the shape of the integrand accurately.

3. **MCMC (`:mcmc`):** The MCMC solver is ideal for dealing with a bundle of integrands that are too large to be computed all at once. It uses the Metropolis-Hastings algorithm to traverse between different integrals, evaluating only one integrand at each step. Though it can be less efficient due to the integral-jumping auto-correlations, it stands out in its ability to handle extremely high-dimensional integrals where other two solvers fail.

Given its robustness and efficiency, the default solver in this package is the `:vegasmc`. To choose a specific solver, use the `solver` parameter in the `integrate` function, like `solver=:vegas`.

Please note that the calling convention for the user-defined integrand for `:mcmc` is slightly different from that of `:vegas` and `:vegasmc`. Please refer to the separate detailed note on this.

Packed variables can enhance the efficiency of :vegasmc and :mcmc solvers by reducing the auto-correlation time of the Markov chain, leading to a more effective sampling proces

## Variables

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

## Parallelization
MCIntegration supports both MPI and multi-thread parallelization. You can even mix them if necessary.

### MPI
To run your code in MPI mode, simply use the command,
```bash
mpiexec -n #NCPU julia ./your_script.jl
```
where `#NCPU` is the number of workers. Internally, the MC sampler will send the blocks (controlled by the argument `Nblock`, see above example code) to different workers, then collect the estimates in the root node. 

Note that you need to install the package [MPI.jl](https://github.com/JuliaParallel/MPI.jl) to use the MPI mode. See this [link](https://juliaparallel.github.io/MPI.jl/stable/configuration/) for the instruction on the configuration.

The user essentially doesn't need to write additional code to support the parallelization. The only tricky part is the output: only the function `MCIntegratoin.integrate` of the root node returns meaningful estimates, while other workers simply returns `nothing`.

### Multi-threading

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
