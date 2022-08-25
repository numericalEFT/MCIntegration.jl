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
julia> X=Continuous(0.0, 1.0) #Define the types variables, the first two arguments set the boundary. see the section [variable](#variable) for more details.
Adaptive continuous variable ∈ [0.0, 1.0). Learning rate = 2.0. 

julia> integrate(c->(X=c.var[1]; X[1]^2+X[2]^2); var = (X, ), dof = [(2, ),])
==================================     Integral-1    ==============================================
  iter          integral                            wgt average                          chi2/dof
---------------------------------------------------------------------------------------------------
     1       0.65074464 ± 0.047564903             0.65074464 ± 0.047564903                 0.0000
     2       0.76468743 ± 0.045398931             0.71036936 ± 0.032840924                 3.0029
     3       0.66119309 ± 0.042882335             0.69218964 ± 0.026073195                 1.9159
     4       0.74476052 ± 0.045914954             0.70500827 ± 0.022672658                 1.6077
     5       0.69634932 ± 0.058623829             0.70388163 ± 0.021146279                 1.2105
     6        0.6543088 ± 0.048101827             0.69585279 ± 0.019358254                 1.1464
     7       0.67531808 ± 0.044556231             0.69259211 ± 0.017754917                 0.9851
     8       0.59976994 ± 0.023144883             0.65820474 ± 0.014087327                 2.2909
     9       0.64934182 ± 0.040158383              0.6572336 ± 0.013293146                 2.0100
    10       0.67476284 ± 0.029880299             0.66012976 ± 0.012145466                 1.8185
---------------------------------------------------------------------------------------------------
Integral-1 = 0.6601297573053011 ± 0.012145466472831361
```

The following is a more involved example with a script.

```julia
using MCIntegration

# Define the integrand 
function integrand(config)
    #config.var is a tuple of variable types specified in the second argument of `MCIntegration.Configuration(...)`
    X = config.var[1]
    if config.curr == 1 #config.curr is the index of the currently sampled integral by MC
        return (X[1]^2 + X[2]^2 < 1.0) ? 1.0 : 0.0
    else
        return (X[1]^2 + X[2]^2 + X[3]^2 < 1.0) ? 1.0 : 0.0
    end
end

# Define how many (degrees of freedom) variables of each type. 
# For example, [[n1, n2], [m1, m2], ...] means the first integral involves n1 varibales of type 1, and n2 variables of type2, while the second integral involves m1 variables of type 1 and m2 variables of type 2. 
dof = [[2,], [3,]]

# perform MC integration. Set print>=0 to print more information.
result = integrate(integrand; 
    var = (Continuous(0.0, 1.0),), 
    dof = dof
    neval=1e4,  # number of integrand evaluation
    niter=10,   # number of iteration.  After each iteraction, the program will try to improve the important sampling
    print=-1    #-1 to not print anything, 0 to print progressbar, >0 to print out internal configurations for every "print" seconds
    )

# In MPI mode, only the root node return meaningful estimates. All other workers simply return nothing
if isnothing(result) == false
    # MCIntegration.summary(result) # uncomment this line to print the summary of the result
    avg, err = result.mean, result.stdev
    println("Circle area: $(avg[1]) +- $(err[1]) (exact: $(π / 4.0))")
    println("Sphere volume: $(avg[2]) +- $(err[2]) (exact: $(4.0 * π / 3.0 / 8))")
end
```

# Variables

This package defines some common types of variables. Internally, each variable type holds a vector of variables (which is the field named `data`). The actual number of variables in this vector is called the degrees of freedom (dof). Note that different integral may share the same variable types, but have different degrees of freedom. In the above code example, the integral for the circle area and the sphere volume both involve the variable type `Continuous`. The former has dof=2, while the latter has dof=3. 

Here we list some of the common variables types

- Continous(lower::Float64, upper::Float64): continuous real-valued variables on the domain [lower, upper). MC will learn the distribution and perform an imporant sampling accordingly.

- Discrete(lower::Int, upper::Int): integer variables in the closed set [lower, upper]. MC will learn the distribution and perform an imporant sampling accordingly.

More supported variables types can be found in the [source code](src/variable.jl).

# Parallelization

MCIntegration supports MPI parallelization. To run your code in MPI mode, simply use the command
```bash
mpiexec julia -n #NCPU ./your_script.jl
```
where `#CPU` is the number of workers. Internally, the MC sampler will send the blocks (controlled by the argument `Nblock`, see above example code) to different workers, then collect the estimates in the root node. 

Note that you need to install the package [MPI.jl](https://github.com/JuliaParallel/MPI.jl) to use the MPI mode. See this [link](https://juliaparallel.github.io/MPI.jl/stable/configuration/) for the instruction on the configuration.

The user essentially doesn't need to write additional code to support the parallelization. The only tricky part is the output: only the function `MCIntegratoin.sample` of the root node returns meaningful estimates, while other workers simply returns `nothing`. 

# Algorithm
The internal algorithm and some simple benchmarks can be found in the [document](docs/src/man/important_sampling.md).

# Q&A

- Q: What if the integral result makes no sense?

  A: One possible reason is the reweight factor. It is important for the Markov chain to visit the integrals with the similar frequency. However, the weight of different integrals may be order-of-magnitude different. It is thus important to reweight the integrals. Internally, the MC sampler try to reweight for each iteration. However, it could fail either 1) the total MC steps is too small so that reweighting doesn't have enough time to show up; ii) the integrals are simply too different, and the internal reweighting subroutine is not smart enough to figure out such difference. If 1) is the case, one either increase the neval. If 2) is the case, one may mannually provide an array of reweight factors when initializes the `MCIntegration.configuration` struct. More details can be found in the [source code](src/variable.jl). 



