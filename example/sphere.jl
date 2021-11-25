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

# Define how to measure the observable
function measure(config)
    factor = 1.0 / config.reweight[config.curr]
    weight = integrand(config)
    config.observable[config.curr] += weight / abs(weight) * factor #note that config.observable is an array with two elements as discussed below
end

# MC step of each block
const blockStep = 1e6

# Define the types variables, the first argument sets the range, the second argument gives the largest change to the variable in one MC update. see the section [variable](#variable) for more details.
T = MCIntegration.Continuous([0.0, 1.0], 0.5)

# Define how many (degrees of freedom) variables of each type. 
# For example, [[n1, n2], [m1, m2], ...] means the first integral involves n1 varibales of type 1, and n2 variables of type2, while the second integral involves m1 variables of type 1 and m2 variables of type 2. 
dof = [[2,], [3,]]

# Define the container for the observable. It must be a number or an array-like object. In this case, the observable has two elements, corresponds to the results for the two integrals. 
obs = [0.0, 0.0]

# Define the configuration struct which is container of all kinds of internal data for MC,
# the second argument is a tuple listing all types of variables, one then specify the degrees of freedom of each variable type in the third argument.  
config = MCIntegration.Configuration(blockStep, (T,), dof, obs)

# perform MC integration. Nblock is the number of independent blocks to estimate the error bar. In MPI mode, the blocks will be sent to different workers. Set "print=n" to control the level of information to print.
avg, err = MCIntegration.sample(config, integrand, measure; Nblock = 64, print = 1)

#avg, err are the same shape as obs. In MPI mode, only the root node return meaningful estimates. All other workers simply return nothing
if isnothing(avg) == false
    println("Circle area: $(avg[1]) +- $(err[1]) (exact: $(π / 4.0))")
    println("Sphere volume: $(avg[2]) +- $(err[2]) (exact: $(4.0 * π / 3.0 / 8))")
end