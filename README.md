# MCIntegration

Universal Monte Carlo calculator for high-dimensional integral with different types of variables.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://numericalEFT.github.io/MCIntegration.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://numericalEFT.github.io/MCIntegration.jl/dev)
[![Build Status](https://github.com/numericalEFT/MCIntegration.jl/workflows/CI/badge.svg)](https://github.com/numericalEFT/MCIntegration.jl/actions)
[![Coverage](https://codecov.io/gh/numericalEFT/MCIntegration.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/numericalEFT/MCIntegration.jl)

MCIntegration provides a Monte Carlo algorithm to calculate high-dimensional integrals that depend on two or more different types of variables (such as momentum vectors, frequencies, and so on). MCIntegration.jl allows the user to choose different important sampling algorithms to efficiently sample different types of variables, which is a huge advantage compared to the commonly used Vegas algorithm:

# Quick start

The following example demonstrates the basic usage of this package. This code calculates the area of a circle and the volume of a sphere using one Markov chain. The code can be found [here](example/sphere.jl).

The following command evaluate a two-dimensional integral ∫dx₁dx₂ (x₁^2+x₂^2) in the domain [0, 1)x[0, 1].
```julia
julia> X=Continuous(0.0, 1.0) #Create a pool of continuous variables. It supports as much as 16 same type of variables. see the section [variable](#variable) for more details.
Adaptive continuous variable in the domain [0.0, 1.0). Max variable number = 16. Learning rate = 2.0.

julia> integrate((X, c)->(X[1]^2+X[2]^2); var = X, dof = 2); 
# We use two of the continuous variables to calculate a two-dimensional integral ∫dx₁dx₂(x₁²+x₂²) in the domain [0, 1)x[0, 1). The parameter c in the integrand is a Configuration struct that carries the internal MC state. See docs for more detail.
==================================     Integral 1    ==============================================
  iter          integral                            wgt average                          chi2/dof
---------------------------------------------------------------------------------------------------
     1       0.73654787 ± 0.055229117             0.73654787 ± 0.055229117                 0.0000
     2       0.78941426 ± 0.055157194             0.76301551 ± 0.03902743                  0.4587
     3       0.61901138 ± 0.037712078             0.68854587 ± 0.027119558                 3.7497
     4       0.56745577 ± 0.036761374             0.64587037 ± 0.021823618                 4.8419
     5       0.76823135 ± 0.044066048             0.66997078 ± 0.01955667                  5.1793
     6       0.69963303 ± 0.038825448             0.67597367 ± 0.017466038                 4.2366
     7         0.661647 ± 0.040428194             0.67372024 ± 0.016033698                 3.5481
     8       0.69492029 ± 0.0346462               0.67745974 ± 0.014551045                 3.0853
     9       0.71363189 ± 0.068011508             0.67904303 ± 0.014229025                 2.7335
    10       0.75550931 ± 0.045812645             0.68577053 ± 0.013588682                 2.7121
---------------------------------------------------------------------------------------------------
Integral 1 = 0.6857705325654451 ± 0.013588681837082542   (chi2/dof = 2.71)
```
By default, the function performs 10 iterations and each iteraction costs about `1e5` evaluations. You may reset these values with `niter` and `neval` keywords arguments.

Internally, the `integrate` function optimizes the important sampling after each iteration. The results generally improves with iteractions. As long as `neval` is sufficiently large, the estimations from different iteractions should be statistically independent. This will justify an average of different iterations weighted by the inverse variance. The assumption of statically independence can be explicitly checked with chi-square test, namely `chi2/dof` should be about one. 

You can also choose different Monte Carlo algorithms by specifing the keyword arguemnt `solver`. By default, the Vegas algorithm with `solver = :vegas` is used. In addition, this package provides two Markov-chain Monte Carlo algorithms for numerical integration. You can call them with `solver = :vegasmc` or `solver = :mcmc`. Check the Algorithm section for more details.

If you have multiple integrals the same set of variables, simply use:
```julia
julia> integrate((X, c)->(X[1]^2+X[2]^2, X[1]^2+X[2]^2+X[3]^2); var = X, dof = [[2,],[3,]], print=-1) # print controls the amount of information to print
dof = [(2,), (3,)]
Integral 1 = 0.664810806792709 ± 0.000793999167254061   (chi2/dof = 0.903)
Integral 2 = 0.9991886776671396 ± 0.000530301553703244   (chi2/dof = 1.75)
```
Here `dof` defines how many (degrees of freedom) variables of each type. For example, [[n1, n2], [m1, m2], ...] means the first integral involves n1 varibales of type 1, and n2 variables of type2, while the second integral involves m1 variables of type 1 and m2 variables of type 2. 

You can also use the julia do-syntax to simplify the integration part in above example:
```julia
julia> integrate(var = (Continuous(0.0, 1.0),), dof = [[2,], [3,]], neval = 1e5, niter = 10, print = -1) do X, c
           r1 = (X[1]^2 + X[2]^2 < 1.0) ? 1.0 : 0.0
           r2 = (X[1]^2 + X[2]^2 + X[3]^2 < 1.0) ? 1.0 : 0.0
           return (r1, r2)
       end
Integral 1 = 0.7858137468685643 ± 0.0003890543762596982   (chi2/dof = 1.63)
Integral 2 = 0.5240009569393371 ± 0.00039066497807783214   (chi2/dof = 0.715)
```

# Variables

The integrals you want to evaluate may have different degrees of freedom, but are probably share the same types of variables. 
In the above code example, the integral for the circle area and the sphere volume both involve the variable type `Continuous`. The former has dof=2, while the latter has dof=3. 

To evaluate these integrals simultaneouly, it makes sense to create a pool of variables. A pool of two common variables types can created with the following constructors:

- Continous(lower::Float64, upper::Float64): continuous real-valued variables on the domain [lower, upper). MC will optimize the distribution and perform an imporant sampling accordingly.
- Discrete(lower::Int, upper::Int): integer variables in the closed set [lower, upper]. MC will learn the distribution and perform an imporant sampling accordingly.

The size of pool can be specified by an optional arguments `size`, which is $16$ by default. Once created, you can access to a given variable with stanard vector indexing interface.
You only need to choose some of them to evaluate a given integral. The others serve as dummy variables. They will not cause any computational overhead.  

After each iteration, the code will try to optimize how the variables are sampled, so that the most important regimes of the integrals will be sampled most frequently. 

More supported variables types can be found in the [source code](src/variable.jl).

# Parallelization

MCIntegration supports MPI parallelization. To run your code in MPI mode, simply use the command
```bash
mpiexec julia -n #NCPU ./your_script.jl
```
where `#CPU` is the number of workers. Internally, the MC sampler will send the blocks (controlled by the argument `Nblock`, see above example code) to different workers, then collect the estimates in the root node. 

Note that you need to install the package [MPI.jl](https://github.com/JuliaParallel/MPI.jl) to use the MPI mode. See this [link](https://juliaparallel.github.io/MPI.jl/stable/configuration/) for the instruction on the configuration.

The user essentially doesn't need to write additional code to support the parallelization. The only tricky part is the output: only the function `MCIntegratoin.integrate` of the root node returns meaningful estimates, while other workers simply returns `nothing`. 

# Algorithm
The internal algorithm and some simple benchmarks can be found in the [document](docs/src/man/important_sampling.md).

# Q&A

- Q: What if the integral result makes no sense?

  A: One possible reason is the reweight factor. It is important for the Markov chain to visit the integrals with the similar frequency. However, the weight of different integrals may be order-of-magnitude different. It is thus important to reweight the integrals. Internally, the MC sampler try to reweight for each iteration. However, it could fail either 1) the total MC steps is too small so that reweighting doesn't have enough time to show up; ii) the integrals are simply too different, and the internal reweighting subroutine is not smart enough to figure out such difference. If 1) is the case, one either increase the neval. If 2) is the case, one may mannually provide an array of reweight factors when initializes the `MCIntegration.configuration` struct. More details can be found in the [source code](src/variable.jl). 



