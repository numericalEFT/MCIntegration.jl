"""
Demostrate do syntax for Monte Carlo simulation.
"""
function Sphere1(neval, alg)
    T = Continuous(0.0, 1.0)
    integrate(var=(T,), dof=[[2,],], neval=neval, block=64, print=-1, solver=alg) do c
        X = c.var[1]
        if (X[1]^2 + X[2]^2 < 1.0)
            return 1.0
        else
            return 0.0
        end
    end
end

function Sphere2(totalstep; offset=0)
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

function TestSingular1(totalstep, alg)
    #log(x)/sqrt(x), singular in x->0
    return integrate(c -> log(c.var[1][1]) / sqrt(c.var[1][1]); neval=totalstep, print=-1, solver=alg)
end

function TestSingular2(totalstep, alg)
    #1/(1-cos(x)*cos(y)*cos(z))
    return integrate(var=(Continuous(0.0, 1π),), dof=[[3,],], neval=totalstep, print=-1, solver=alg) do config
        x = config.var[1]
        return 1.0 / (1.0 - cos(x[1]) * cos(x[2]) * cos(x[3])) / π^3
    end
end

function TestComplex1(totalstep, alg)
    return integrate(neval=totalstep, print=-1, type=ComplexF64, solver=alg) do config
        x = config.var[1]
        return x[1] + x[1]^2 * 1im
    end
end

function TestComplex2(totalstep, alg)
    return integrate(dof=[[1,], [1,]], neval=totalstep, print=-1, type=ComplexF64, solver=alg) do config
        x = config.var[1]
        #return a (real, complex) 
        #the code should handle real -> complex conversion
        return x[1], x[1]^2 * 1im
    end
end

@testset "MonteCarlo Sampler" begin
    neval = 1000_000

    println("Sphere1")
    check(Sphere1(neval, :MC), π / 4.0)
    # check(Sphere2(neval), π / 4.0)
    println("Sphere2")
    # check(Sphere3(neval), [π / 4.0, 4.0 * π / 3.0 / 8])
    # check(TestDiscrete(neval), 6.0)
    println("Singular1")
    check(TestSingular1(neval, :MC), -4.0)
    println("Singular2")
    check(TestSingular2(neval, :MC), 1.3932)

    neval = 1000_00
    println("Complex1")
    check_complex(TestComplex1(neval, :MC), 0.5 + 1.0 / 3 * 1im)
    println("Complex2")
    check_complex(TestComplex2(neval, :MC), [0.5, 1.0 / 3 * 1im])

end
