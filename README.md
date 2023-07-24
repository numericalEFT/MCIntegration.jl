# MCIntegration.jl: Monte Carlo Integration in Julia

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://numericalEFT.github.io/MCIntegration.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://numericalEFT.github.io/MCIntegration.jl/dev)
[![Build Status](https://github.com/numericalEFT/MCIntegration.jl/workflows/CI/badge.svg)](https://github.com/numericalEFT/MCIntegration.jl/actions)
[![Coverage](https://codecov.io/gh/numericalEFT/MCIntegration.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/numericalEFT/MCIntegration.jl)

## Why Choose MCIntegration.jl?
MCIntegration.jl is a comprehensive Julia package designed to handle both regular and singular high-dimensional integrals with ease. Its implementation of robust Monte Carlo integration methods makes it a versatile tool in various scientific domains, including high-energy physics, material science, computational chemistry, financial mathematics, and machine learning.

The high-level simplicity and flexibility of Julia combined with the performance capabilities of C/C++-like compiled languages make it a fantastic choice for implementing Monte Carlo methods. Monte Carlo methods, which require extensive computations, can greatly benefit from Julia's just-in-time (JIT) compilation that allows `MCIntegration.jl` to perform calculations at a near-C/C++ efficiency. Moreover, the intuitive high-level syntax of Julia allows users to define their integrands effortlessly, adding to the customizability and user-friendliness of `MCIntegration.jl`.

## Installation
To install MCIntegration.jl, use Julia's package manager. Open the Julia REPL, type `]` to enter the package mode, and then:
```
pkg> add MCIntegration
```
## Quick Start
MCIntegration.jl simplifies complex integral calculations. Here are two examples to get you started.

To estimate the integral $\int_0^1 \frac{\log(x)}{\sqrt{x}} dx = 4$, you can use:
```julia
julia> f(x, c) = log(x[1]) / sqrt(x[1])   # Define your integrand
julia> integrate(f, var = Continuous(0, 1), neval=1e5)   # Perform the MC integration for 1e5 steps
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
$\int d\vec{x} \int d\vec{y}... \vec{f}(\vec{x}, \vec{y}...)$
The package handles both continuous and discrete variables; for discrete variables, the integrals are interpreted as summations.

In `MCIntegration.jl`, the **"degree of freedom" (`dof`)** defines the dimensions for each variable group in the integrand. It's represented as a list of dimensions: dof($\vec{f}$) = [[dim($\vec{x}$), dim($\vec{y}$), ...], ...], with dim($\vec{f}$) elements.

The building blocks of variable organization in `MCIntegration.jl` are the **variable vectors** (like $\vec{x}$, $\vec{y}$, ...), which are implemented as **unlimited** pools of variables. These variable vectors can be combined or used individually, depending on the integrand's structure. 

- **Variable Vectors (Symmetric Variables):** If the variables in the integrands are interchangeable, they can be organized into a variable vector acting as a pool. All variables within the same pool are sampled from the same optimized distributions.

- **Composite Variable Vectors:** When two or more variable vectors consistently appear together across all integrands, they can be bundled to form composite variable vectors. Composite variables share the same `dof` and can be updated together during the integration process, offering computational efficiency especially when working with MCMC-based solvers (see the next section for more detail).

Both variable vectors and composite variable vectors can be organized into `Tuple`, offering more complex interactions between different variables or composite variables in the integrands.

Here are examples to illustrate the usage of different types of variable vectors:

- Symmetric Variables (Variable Vector): Estimate π
```julia
julia> f(x, c) = x[1]^2 + x[2]^2 < 1
julia> integrate(f; var = Continuous(-1, 1), dof = [[2, ],]) # dof must be provided for multi-dimensional integrands
Integral 1 = 3.1316915341619413 ± 0.008785871829296759   (reduced chi2 = 0.298)
```
- Composite Variable Vector: Estimate π with polar coordinate (r, θ)
```julia
julia> g((r, θ), c) = r[1] # Unpack the variables into r and θ. The integrand is independent of θ.
julia> integrate(g; var = Continuous([(0, 1), (0, 2π)]), dof = [(1, ),]) 
# Alternatively, use the constructor: CompositeVar(Continuous(0, 1), Continuous(0, 2π))
Integral 1 = 3.14367422926071 ± 0.0011572440016582415   (reduced chi2 = 0.735)
```

- Tuple of Variable Vectors: Calculate $\sum_{n \ge 0} \int_0^1 (-1)^n x^{2n}dx = \pi/4$
```julia
julia> f((n, x), c) = 4*(-1)^n[1]*x[1]^(2*n[1])
julia> integrate(f; var = (Discrete(0, 100), Continuous(0, 1)), dof = [(1, 1),], neval=1e5)
Integral 1 = 3.141746201859978 ± 0.04261519744132012   (reduced chi2 = 0.611)
```

## Selecting Algorithms

MCIntegration.jl offers three Monte Carlo integration algorithms, all of which leverage the Vegas map technique for importance sampling. This approach constructs a piecewise constant Vegas map, a probability distribution function approximating the shape of the integrand to enhance the efficiency of the integral estimation.

Here's a brief overview of the three solvers:

1. **Vegas (`:vegas`):** The classic Vegas algorithm uses the Monte Carlo method and samples all integrands across all variables simultaneously at each step. It is efficient for low-dimensional integrals but might struggle with high-dimensional ones where the Vegas map fails to accurately mimic the integrand's shape.

2. **Vegas with MCMC (`:vegasmc`):** This innovative solver, first introduced in MCIntegration.jl, combines Vegas with Markov-chain Monte Carlo. This hybrid approach provides a robust solution, especially for intricate, high-dimensional integrals. In the Vegas MC approach, a variable is selected randomly, and a Metropolis-Hastings algorithm is utilized to propose a new variable based on the Vegas map. This update is applied simultaneously across all integrands, improving robustness when the Vegas map struggles with approximating the shape of the integrand accurately.

3. **MCMC (`:mcmc`):** The MCMC solver is ideal for dealing with a bundle of integrands that are too large to be computed all at once. It uses the Metropolis-Hastings algorithm to traverse between different integrals, evaluating only one integrand at each step. Though it can be less efficient due to the integral-jumping auto-correlations, it stands out in its ability to handle extremely high-dimensional integrals where other two solvers fail.

Given its robustness and efficiency, the default solver in this package is the `:vegasmc`. To choose a specific solver, use the `solver` parameter in the `integrate` function, like `solver=:vegas`.

Please note that the calling convention for the user-defined integrand for `:mcmc` is slightly different from that of `:vegas` and `:vegasmc`. Please refer to the separate detailed note on this.

Packed variables can enhance the efficiency of :vegasmc and :mcmc solvers by reducing the auto-correlation time of the Markov chain, leading to a more effective sampling process.


## Parallelization

Parallelization is a vital aspect of `MCIntegration.jl`, enhancing the performance of your Monte Carlo simulations. The package supports both MPI and multi-thread parallelization, with an option to combine them as required.

- MPI
  With MPI, you can run your code in a distributed manner, using the command:
  ```bash
  mpiexec -n NCPU julia your_script.jl
  ```
  Here, `NCPU` denotes the number of workers. The MC sampler internally dispatches blocks (controlled by the Nblock argument) to different workers and collects the estimates on the root node. While using MPI, the `integrate` function returns meaningful estimates only for the root node. For other workers, it returns `nothing`.

  **Note:** For MPI functionality, install [MPI.jl](https://github.com/JuliaParallel/MPI.jl) package and follow the [configuration](https://juliaparallel.github.io/MPI.jl/stable/configuration/) instructions.

   

- Multi-threading
  To enable multi-threading, start Julia as follows:
  ```bash
  julia -t NCPU your_script.jl
  ```
  Remember, all threads share the same memory, so ensure your integrand and measure functions are thread-safe. Check Julia's official [documentation](https://docs.julialang.org/en/v1/manual/multi-threading/) for further guidance. For multi-threading, you have two options:
  - **Concurrent Integration:** Each thread independently calls `integrate` to perform separate integrations.
  - **Block-wise Parallelization:** Only the main thread invokes `integrate`, while the computation blocks within are parallelized across multiple threads. To apply this, use `integrate` with the argument `parallel = :thread`.

  The following examples demonstrate the difference between two approaches,
  ```julia
  # Concurrent Integration
  Threads.@threads for i = 1:3
      integrate((x, c) -> x[1]^i, verbose=-2)
  end

  # Block-wise Parallelization
  for i = 1:3
      integrate((x, c) -> x[1]^i, verbose=-2, parallel=:thread)
  end
  ```

## Detailed Examples and Advanced Usage
For more advanced use cases and in-depth tutorials, please see the [tutorial](https://numericaleft.github.io/MCIntegration.jl/dev/#MCIntegration) in the full `MCIntegration.jl` [documentation](https://numericaleft.github.io/MCIntegration.jl/dev/). Examples include handling large sets of integrands, histogram measurement, and user-defined configurations.

## Getting Help
For further information and assistance, please refer to the full `MCIntegration.jl` [documentation](https://numericaleft.github.io/MCIntegration.jl/dev/). If you encounter issues or have further questions, don't hesitate to open an issue on the [GitHub repository](https://github.com/numericalEFT/MCIntegration.jl).

## Acknowledgements and Related Packages
The development of `MCIntegration.jl` has been greatly inspired and influenced by several significant works in the field of numerical integration. We would like to express our appreciation to the following:

- [Cuba](https://feynarts.de/cuba/) and [Cuba.jl](https://github.com/giordano/Cuba.jl): The Cuba library offers numerous Monte Carlo algorithms for multidimensional numerical integration, and Cuba.jl provides a proficient Julia interface to it. While `MCIntegration.jl` is an independent and native Julia package, we acknowledge the foundational contributions of Cuba and Cuba.jl. For further details, refer to the Cuba [homepage](https://feynarts.de/cuba/) and Cuba.jl [documentation](https://giordano.github.io/Cuba.jl/stable/). **Reference: T. Hahn, Comput. Phys. Commun. 168, 78 (2005) [arXiv:hep-ph/0404043](https://arxiv.org/abs/hep-ph/0404043)**.

- [vegas](https://github.com/gplepage/vegas) A Python package offering Monte Carlo estimations of multidimensional integrals, with notable improvements on the original Vegas algorithm. It's been a valuable reference for us. Learn more from the vegas [documentation](https://vegas.readthedocs.io/). **Reference: G. P. Lepage, J. Comput. Phys. 27, 192 (1978) and G. P. Lepage, J. Comput. Phys. 439, 110386 (2021) [arXiv:2009.05112](https://arxiv.org/abs/2009.05112)**. 

These groundbreaking efforts have paved the way for our project. We extend our deepest thanks to their creators and maintainers.