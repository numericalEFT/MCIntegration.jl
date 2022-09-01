"""
Demostrate do syntax for Monte Carlo simulation.
"""
function Sphere1(neval)
    T = Continuous(0.0, 1.0)
    integrate(var=(T,), dof=[[2,],], neval=neval, block=64, print=-1) do c
        X = c.var[1]
        if (X[1]^2 + X[2]^2 < 1.0)
            return 1.0
        else
            return 0.0
        end
    end
end

# function Sphere2(totalstep)
#     function integrand(config)
#         X = config.var[1]
#         if (X[1][1]^2 + X[1][2]^2 < 1.0)
#             return 1.0
#         else
#             return 0.0
#         end
#     end

#     T = Dist.TauPair(1.0, 1.0 / 2.0)
#     dof = [[1,],] # number of T variable for the normalization and the integrand
#     config = Configuration(var=(T,), dof=dof)
#     @inferred integrand(config) #make sure the type is inferred for the integrand function
#     return integrate(integrand, config=config, neval=totalstep, block=64, print=-1)
# end

function Sphere3(totalstep; offset=0)
    function integrand(config)
        @assert config.curr == 1 || config.curr == 2 "$(config.curr) is not a valid integrand"
        X = config.var[1]
        if config.curr == 1
            if (X[1+offset]^2 + X[2+offset]^2 < 1.0)
                return 1.0
            else
                return 0.0
            end
        else
            if (X[1+offset]^2 + X[2+offset]^2 + X[3+offset]^2 < 1.0)
                return 1.0
            else
                return 0.0
            end
        end
    end

    function measure(config)
        config.observable[config.curr] += config.relativeWeight
    end

    T = Continuous(0.0, 1.0; offset=offset)
    dof = [[2,], [3,]] # number of T variable for the normalization and the integrand
    config = Configuration(var=(T,), dof=dof, obs=[0.0, 0.0]; neighbor=[(1, 3), (1, 2)])
    @inferred integrand(config) #make sure the type is inferred for the integrand function
    return integrate(integrand, measure=measure, config=config, neval=totalstep, block=64, print=-1)
end

# function Exponential1(totalstep)
#     function integrand(config)
#         X = config.var[1]
#         return exp(-X[1])
#     end

#     function measure(config)
#         factor = 1.0 / config.reweight[config.curr]
#         weight = integrand(config)
#         config.observable += weight / abs(weight) * factor
#     end

#     K = MCIntegration.RadialFermiK(1.0, 0.01)
#     dof = [[1,],] # number of T variable for the normalization and the integrand
#     config = MCIntegration.Configuration((K,), dof, 0.0)
#     @inferred integrand(config) #make sure the type is inferred for the integrand function
#     result = MCIntegration.sample(config, integrand, measure; neval=totalstep, block=64, print=-1)
#     # avg, err = MonteCarlo.sample(totalstep, (T,), dof, [0.0, ], integrand, measure; Nblock=64, print=-1)
#     return result.mean, result.stdev
# end
# const LorentzN = 1000

# function Lorentz1(totalstep)
#     function integrand(config)
#         X = config.var[1]
#         return 1.0 / ((X[1] - 1)^2 + (1.0 / LorentzN)^2)
#     end

#     function measure(config)
#         factor = 1.0 / config.reweight[config.curr]
#         weight = integrand(config)
#         config.observable += weight / abs(weight) * factor
#     end

#     K = MCIntegration.RadialFermiK(1.0, 0.001)
#     dof = [[1,],] # number of T variable for the normalization and the integrand
#     config = MCIntegration.Configuration((K,), dof, 0.0)
#     @inferred integrand(config) #make sure the type is inferred for the integrand function
#     result = MCIntegration.sample(config, integrand, measure; neval=totalstep, block=64, print=-1)
#     # avg, err = MonteCarlo.sample(totalstep, (T,), dof, [0.0, ], integrand, measure; Nblock=64, print=-1)
#     return result.mean, result.stdev
# end


# function Exponential2(totalstep)
#     function integrand(config)
#         @assert config.curr == 1 || config.curr == 2
#         X = config.var[1]
#         if config.curr == 1
#             return exp(-X[1])
#         else
#             return exp(-X[1] - X[2])
#         end
#     end

#     function measure(config)
#         factor = 1.0 / config.reweight[config.curr]
#         weight = integrand(config)
#         config.observable[config.curr] += weight / abs(weight) * factor
#     end

#     K = MCIntegration.RadialFermiK(1.0, 0.01)
#     dof = [[1,], [2,]] # number of T variable for the normalization and the integrand
#     config = MCIntegration.Configuration((K,), dof, [0.0, 0.0])
#     @inferred integrand(config) #make sure the type is inferred for the integrand function
#     result = MCIntegration.sample(config, integrand, measure; neval=totalstep, block=64, print=-1)
#     # avg, err = MonteCarlo.sample(totalstep, (T,), dof, [0.0, ], integrand, measure; Nblock=64, print=-1)
#     return result.mean, result.stdev
# end


# function Lorentz2(totalstep)
#     function integrand(config)
#         @assert config.curr == 1 || config.curr == 2
#         X = config.var[1]
#         if config.curr == 1
#             return 1.0 / ((X[1] - 1)^2 + (1.0 / LorentzN)^2)
#         else
#             return 1.0 / ((X[1] - 1)^2 + (1.0 / LorentzN)^2) * exp(-X[2])
#         end
#     end

#     function measure(config)
#         factor = 1.0 / config.reweight[config.curr]
#         weight = integrand(config)
#         config.observable[config.curr] += weight / abs(weight) * factor
#     end

#     K = MCIntegration.RadialFermiK(1.0, 0.01)
#     dof = [[1,], [2,]] # number of T variable for the normalization and the integrand
#     config = MCIntegration.Configuration((K,), dof, [0.0, 0.0])
#     @inferred integrand(config) #make sure the type is inferred for the integrand function
#     result = MCIntegration.sample(config, integrand, measure; neval=totalstep, block=64, print=-1)
#     # avg, err = MonteCarlo.sample(totalstep, (T,), dof, [0.0, ], integrand, measure; Nblock=64, print=-1)
#     return result.mean, result.stdev
# end

function TestDiscrete(totalstep)
    function integrand(config)
        x = config.var[1][1]
        return x
    end

    X = Discrete(1, 3, adapt=true)
    dof = [[1,],] # number of X variable of the integrand
    config = Configuration(var=(X,), dof=dof)
    @inferred integrand(config) #make sure the type is inferred for the integrand function
    return integrate(integrand; config=config, neval=totalstep, niter=10, block=64, print=-1)
end

function TestSingular1(totalstep)
    #log(x)/sqrt(x), singular in x->0
    return integrate(c -> log(c.var[1][1]) / sqrt(c.var[1][1]); neval=totalstep, print=-1)
end

function TestSingular2(totalstep)
    #1/(1-cos(x)*cos(y)*cos(z))
    return integrate(var=(Continuous(0.0, 1π),), dof=[[3,],], neval=totalstep, print=-1) do config
        x = config.var[1]
        return 1.0 / (1.0 - cos(x[1]) * cos(x[2]) * cos(x[3])) / π^3
    end
end

function TestComplex1(totalstep)
    return integrate(neval=totalstep, print=-1, type=ComplexF64) do config
        x = config.var[1]
        return x[1] + x[1]^2 * 1im
    end
end

function TestComplex2(totalstep)
    return integrate(dof=[[1,], [1,]], neval=totalstep, print=-1, type=ComplexF64) do config
        x = config.var[1]
        #return a (real, complex) 
        #the code should handle real -> complex conversion
        return x[1], x[1]^2 * 1im
    end
end

@testset "MonteCarlo Sampler" begin
    neval = 1000_000

    println("Sphere1")
    # check(Sphere1(neval), π / 4.0)
    # check(Sphere2(neval), π / 4.0)
    println("Sphere3")
    # check(Sphere3(neval), [π / 4.0, 4.0 * π / 3.0 / 8])
    # check(TestDiscrete(neval), 6.0)
    println("Singular1")
    check(TestSingular1(neval), -4.0)
    println("Singular2")
    check(TestSingular2(neval), 1.3932)

    neval = 1000_00
    println("Complex1")
    check_complex(TestComplex1(neval), 0.5 + 1.0 / 3 * 1im)
    println("Complex2")
    check_complex(TestComplex2(neval), [0.5, 1.0 / 3 * 1im])

end
