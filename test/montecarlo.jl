"""
Demostrate do syntax for Monte Carlo simulation.
"""
function Sphere1(neval, alg)
    X = Continuous(0.0, 1.0)
    integrate(var=(X,), dof=[[2,],], neval=neval, block=64, print=-1, solver=alg) do X
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

function Sphere3(totalstep, alg; offset=0)
    function integrand(X)
        i1 = (X[1+offset]^2 + X[2+offset]^2 < 1.0) ? 1.0 : 0.0
        i2 = (X[1+offset]^2 + X[2+offset]^2 + X[3+offset]^2 < 1.0) ? 1.0 : 0.0
        return i1, i2
    end

    function measure(obs, weights)
        obs[1] += weights[1]
        obs[2] += weights[2]
    end

    T = Continuous(0.0, 1.0; offset=offset)
    dof = [[2,], [3,]] # number of T variable for the normalization and the integrand
    config = Configuration(var=(T,), dof=dof, obs=[0.0, 0.0]; neighbor=[(1, 3), (1, 2)])
    # @inferred integrand(config) #make sure the type is inferred for the integrand function
    return integrate(integrand, measure=measure, config=config, neval=totalstep, block=64, print=-1, solver=alg)
end

function TestDiscrete(totalstep, alg)
    X = Discrete(1, 3, adapt=true)
    dof = [[1,],] # number of X variable of the integrand
    config = Configuration(var=(X,), dof=dof)
    return integrate(X -> X[1]; config=config, neval=totalstep, niter=10, block=64, print=-1, solver=alg)
end

function TestSingular1(totalstep, alg)
    #log(x)/sqrt(x), singular in x->0
    return integrate(X -> log(X[1]) / sqrt(X[1]); neval=totalstep, print=-1, solver=alg)
end

function TestSingular2(totalstep, alg)
    #1/(1-cos(x)*cos(y)*cos(z))
    return integrate(var=(Continuous(0.0, 1π),), dof=[[3,],], neval=totalstep, print=-1, solver=alg) do x
        return 1.0 / (1.0 - cos(x[1]) * cos(x[2]) * cos(x[3])) / π^3
    end
end

function TestComplex1(totalstep, alg)
    return integrate(neval=totalstep, print=-1, type=ComplexF64, solver=alg) do x
        return x[1] + x[1]^2 * 1im
    end
end

function TestComplex2(totalstep, alg)
    return integrate(dof=[[1,], [1,]], neval=totalstep, print=-1, type=ComplexF64, solver=alg) do x
        #return a tuple (real, complex) 
        #the code should handle real -> complex conversion
        return x[1], x[1]^2 * 1im
    end
end

@testset "Vegas Sampler" begin
    neval = 1000_00

    println("Sphere 2D")
    check(Sphere1(neval, :vegas), π / 4.0)
    # check(Sphere2(neval), π / 4.0)
    println("Sphere 2D + 3D")
    check(Sphere3(neval, :vegas), [π / 4.0, 4.0 * π / 3.0 / 8])
    println("Discrete")
    check(TestDiscrete(neval, :vegas), 6.0)
    println("Singular1")
    res = TestSingular1(neval, :vegas)
    check(res, -4.0)
    @test res.stdev[1] < 0.0004 #make there is no regression, vegas typically gives accuracy ~0.0002 with 1e5x10 evaluations
    println("Singular2")
    check(TestSingular2(neval, :vegas), 1.3932)

    neval = 1000_00
    println("Complex1")
    check_complex(TestComplex1(neval, :vegas), 0.5 + 1.0 / 3 * 1im)
    println("Complex2")
    check_complex(TestComplex2(neval, :vegas), [0.5, 1.0 / 3 * 1im])

end

@testset "Markov-Chain Vegas" begin
    neval = 1000_00

    # TODO: so far vegas MC doesn't work with Sphere1 and Sphere2. These integrals vanishes in some regimes, making the measurement of the normalization integral unreliable.
    # println("Sphere1")
    # check(Sphere1(neval, :MCMC), π / 4.0)
    # check(Sphere2(neval), π / 4.0)
    # println("Sphere2")
    # check(Sphere3(neval), [π / 4.0, 4.0 * π / 3.0 / 8])
    println("Discrete")
    check(TestDiscrete(neval, :vegasmc), 6.0)
    println("Singular1")
    res = TestSingular1(neval, :vegasmc)
    check(res, -4.0)
    # @test res.stdev[1] < 0.0004 #make there is no regression, vegas typically gives accuracy ~0.0002 with 1e5x10 evaluations
    println("Singular2")
    check(TestSingular2(neval, :vegasmc), 1.3932)

    neval = 1000_00
    println("Complex1")
    check_complex(TestComplex1(neval, :vegasmc), 0.5 + 1.0 / 3 * 1im)
    println("Complex2")
    check_complex(TestComplex2(neval, :vegasmc), [0.5, 1.0 / 3 * 1im])

end
