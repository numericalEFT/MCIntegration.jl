# MCIntegration

Universal Monte Carlo calculator for high-dimensional integral

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://numericalEFT.github.io/MCIntegration.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://numericalEFT.github.io/MCIntegration.jl/dev)
[![Build Status](https://github.com/numericalEFT/MCIntegration.jl/workflows/CI/badge.svg)](https://github.com/numericalEFT/MCIntegration.jl/actions)
[![Coverage](https://codecov.io/gh/numericalEFT/MCIntegration.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/numericalEFT/MCIntegration.jl)

MCIntegration is Monte Carlo calculator for generic high-dimensional integral. 

# Quick start

The following example demonstrates the basic usage of this package. This code calculates the are of a circle and the volume of a sphere using one Markov chain. The code is adapated from one of the [test cases](test/montecarlo.jl).

```julia
using MCIntegration

# Define the integrand 
function integrand(config)
    @assert config.curr == 1 || config.curr == 2
    X = config.var[1]
    if config.curr == 1
        if (X[1]^2 + X[2]^2 < 1.0) 
            return 1.0 
        else 
            return 0.0
        end
    else
        if (X[1]^2 + X[2]^2 + X[3]^2 < 1.0) 
            return 1.0 
        else 
            return 0.0
        end
    end
end

# Define how to measure the observable
function measure(config)
    factor = 1.0 / config.reweight[config.curr]
    weight = integrand(config)
    config.observable[config.curr] += weight / abs(weight) * factor #note that config.observable is an array with two elements as discussed below
end

# Define the types variables, see the section [variable](#variable) for more details.
T = MCIntegration.Tau(1.0, 1.0 / 2.0) 

# Define how many (degrees of freedom) variables of each type. 
# For example, [[n1, n2], [m1, m2], ...] means the first integral involves n1 varibales of type 1, and n2 variables of type2, while the second integral involves m1 variables of type 1 and m2 variables of type 2. 
dof = [[2, ], [3, ]] 

# Define the container for the observable. It must be a number or an array-like object. In this case, the observable has two elements, corresponds to the results for the two integrals. 
obs = [0.0, 0.0]

# Define the configuration variable, the second argument is a tuple listing all types of variables, one then specify the degrees of freedom of each variable type in the third argument.  
config = MCIntegration.Configuration(totalstep, (T,), dof, obs)

# perform MC integration. Nblock is the number of independent blocks to estimate the error bar. In MPI mode, the blocks will be sent to different works. Set "print=n" to control the level of information to print.
avg, err = MCIntegration.sample(config, integrand, measure; Nblock=64, print=1)

println("Estimation: $avg +- $err")
```