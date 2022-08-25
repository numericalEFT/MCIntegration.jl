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
    var=(Continuous(0.0, 1.0),),
    dof=dof,
    neval=1e6,  # number of integrand evaluation
    niter=10,   # number of iteration.  After each iteraction, the program will try to improve the important sampling
    print=0     #-1 to not print anything, 0 to print progressbar, >0 to print out internal configurations for every "print" seconds
)

# In MPI mode, only the root node return meaningful estimates. All other workers simply return nothing
if isnothing(result) == false
    # MCIntegration.summary(result) # uncomment this line to print the summary of the result
    avg, err = result.mean, result.stdev
    println("Circle area: $(avg[1]) +- $(err[1]) (exact: $(π / 4.0))")
    println("Sphere volume: $(avg[2]) +- $(err[2]) (exact: $(4.0 * π / 3.0 / 8))")
end