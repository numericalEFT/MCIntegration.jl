"""
Demostrate do syntax for Monte Carlo simulation.
"""
function Sphere1(neval, alg)
    X = Continuous(0.0, 1.0)
    f(x, c) = (x[1]^2 + x[2]^2 < 1.0) ? 1.0 : 0.0
    f(idx, x, c)::Float64 = f(x, c)
    return integrate(f; var=(X,), dof=[[2,],], neval=neval, print=-1, solver=alg)
end

"""
Unit test for MCMC integration with reweighting goals specified.
"""
function TestMCMCReweight(neval)
    X = Continuous(0.0, 1.0)
    return integrate((idx, x, c) -> 1.0; var=(X,), dof=[[1,],], neval=neval, print=-1, solver=:mcmc, reweight_goal=ones(2))
end

function Sphere2(totalstep, alg; offset=0)
    function integrand(X, config) # return a tuple of two integrands
        i1 = (X[1+offset]^2 + X[2+offset]^2 < 1.0) ? 1.0 : 0.0
        i2 = (X[1+offset]^2 + X[2+offset]^2 + X[3+offset]^2 < 1.0) ? 1.0 : 0.0
        return i1, i2
    end
    function integrand(idx, X, config) # return one of the integrand
        @assert idx == 1 || idx == 2 "$(idx) is not a valid integrand"
        if idx == 1
            return (X[1+offset]^2 + X[2+offset]^2 < 1.0) ? 1.0 : 0.0
        else
            return (X[1+offset]^2 + X[2+offset]^2 + X[3+offset]^2 < 1.0) ? 1.0 : 0.0
        end
    end

    function measure(X, obs, relativeWeights, config)
        obs .+= relativeWeights
    end
    function measure(idx, X, obs, relativeWeight, config)
        obs[idx] += relativeWeight
    end

    T = Continuous(0.0, 1.0; offset=offset)
    # dof = [2 3] # a 1x2 matrix, each row is the number of dof for each integrand
    dof = [[2,], [3,]] # a 1x2 matrix, each row is the number of dof for each integrand
    config = Configuration(var=(T,), dof=dof; neighbor=[(1, 3), (1, 2)])
    @inferred integrand(config.var[1], config) #make sure the type is inferred for the integrand function
    @inferred integrand(1, config.var[1], config) #make sure the type is inferred for the integrand function
    return integrate(integrand; config=config, neval=totalstep, print=-1, solver=alg, debug=true, measure)
end

# test obs with multiple integrands with different types
function Sphere3(totalstep, alg; offset=0)
    function integrand(X, config) # return a tuple of two integrands
        i1 = (X[1+offset]^2 + X[2+offset]^2 < 1.0) ? 1.0 : 0.0
        i2 = (X[1+offset]^2 + X[2+offset]^2 + X[3+offset]^2 < 1.0) ? 1.0 : 0.0
        return i1, i2
    end
    function integrand(idx, X, config) # return one of the integrand
        @assert idx == 1 || idx == 2 "$(idx) is not a valid integrand"
        if idx == 1
            return (X[1+offset]^2 + X[2+offset]^2 < 1.0) ? 1.0 : 0.0
        else
            return (X[1+offset]^2 + X[2+offset]^2 + X[3+offset]^2 < 1.0) ? 1.0 : 0.0
        end
    end

    function measure(X, obs, relativeWeights, config)
        obs[1] += relativeWeights[1]
        obs[2][1] += relativeWeights[2]
        obs[2][2] += relativeWeights[2] * 2.0
    end
    function measure(idx, X, obs, relativeWeight, config)
        if idx == 1
            obs[idx] += relativeWeight
        elseif idx == 2
            obs[idx][1] += relativeWeight
            obs[idx][2] += relativeWeight * 2.0
        else
            error("invalid idx: $(idx)")
        end
    end

    T = Continuous(0.0, 1.0; offset=offset)
    # dof = [2 3] # a 1x2 matrix, each row is the number of dof for each integrand
    dof = [[2,], [3,]] # a 1x2 matrix, each row is the number of dof for each integrand
    obs = [0.0, [0.0, 0.0]]
    config = Configuration(var=(T,), dof=dof; neighbor=[(1, 3), (1, 2)], obs=obs)
    @inferred integrand(config.var[1], config) #make sure the type is inferred for the integrand function
    @inferred integrand(1, config.var[1], config) #make sure the type is inferred for the integrand function
    return integrate(integrand; config=config, neval=totalstep, print=-1, solver=alg, debug=true, measure)
end

function TestDiscrete(totalstep, alg)
    X = Discrete(1, 3, adapt=true)
    dof = [[1,],] # number of X variable of the integrand
    config = Configuration(var=(X,), dof=dof)
    f(x, c) = x[1]
    f(idx, x, c)::Int = f(x, c)
    return integrate(f; config=config, neval=totalstep, niter=10, print=-1, solver=alg, debug=true)
end

function TestSingular1(totalstep, alg)
    #log(x)/sqrt(x), singular in x->0
    f(X, c) = log(X[1]) / sqrt(X[1])
    f(idx, X, c) = log(X[1]) / sqrt(X[1])
    return integrate(f; neval=totalstep, print=-1, solver=alg)
end

function TestSingular2(totalstep, alg)
    #1/(1-cos(x)*cos(y)*cos(z))
    if alg == :mcmc
        return integrate(var=(Continuous(0.0, 1π),), dof=[[3,],], neval=totalstep, print=-1, solver=alg) do idx, x, c
            return 1.0 / (1.0 - cos(x[1]) * cos(x[2]) * cos(x[3])) / π^3
        end
    else
        return integrate(var=(Continuous(0.0, 1π),), dof=[[3,],], neval=totalstep, print=-1, solver=alg) do x, c
            return 1.0 / (1.0 - cos(x[1]) * cos(x[2]) * cos(x[3])) / π^3
        end
    end
end

function TestSingular2_CompositeVar(totalstep, alg)
    #1/(1-cos(x)*cos(y)*cos(z))
    X, Y, Z = Continuous(0.0, 1π), Continuous(0.0, 1π), Continuous(0.0, 1π)
    C = Dist.CompositeVar(X, Y, Z)
    if alg == :mcmc
        return integrate(var=C, dof=1, neval=totalstep, print=-1, solver=alg) do idx, cvars, c
            x, y, z = cvars
            return 1.0 / (1.0 - cos(x[1]) * cos(y[1]) * cos(z[1])) / π^3
        end
    else
        return integrate(var=C, dof=1, neval=totalstep, print=-1, solver=alg) do cvars, c
            x, y, z = cvars
            return 1.0 / (1.0 - cos(x[1]) * cos(y[1]) * cos(z[1])) / π^3
        end
    end
end

function TestSingular2_Continuous_HighDim(totalstep, alg)
    #1/(1-cos(x)*cos(y)*cos(z))
    C = Continuous([(0.0, 1π), (0.0, 1π), (0.0, 1π)])
    println("compsitevar type: ", typeof(C))
    if alg == :mcmc
        return integrate(var=C, dof=1, neval=totalstep, print=-1, solver=alg) do idx, cvars, c
            x, y, z = cvars
            return 1.0 / (1.0 - cos(x[1]) * cos(y[1]) * cos(z[1])) / π^3
        end
    else
        return integrate(var=C, dof=1, neval=totalstep, print=-1, solver=alg) do cvars, c
            x, y, z = cvars
            return 1.0 / (1.0 - cos(x[1]) * cos(y[1]) * cos(z[1])) / π^3
        end
    end
end

function TestComplex1(totalstep, alg)
    f(x, c) = x[1] + x[1]^2 * 1im
    f(idx, x, c)::ComplexF64 = f(x, c) # dispatch with args seems require type annotation
    integrate(f; neval=totalstep, print=-1, type=ComplexF64, solver=alg, debug=true)
end

function TestComplex2(totalstep, alg)
    function integrand(x, c) #return a tuple (real, complex) 
        #the code should handle real -> complex conversion
        return x[1], x[1]^2 * 1im
    end
    function integrand(idx, x, c) # return one of the integrand
        return idx == 1 ? x[1] + 0im : (x[1]^2 * 1im)
    end
    res = integrate(integrand; dof=[[1,], [1,]], neval=totalstep, print=-1, type=ComplexF64, solver=alg, debug=true)
    config = res.config
    @inferred integrand(config.var[1], config) #make sure the type is inferred for the integrand function
    @inferred integrand(1, config.var[1], config) #make sure the type is inferred for the integrand function
    return res
end

function TestComplex2_inplace(totalstep, alg)
    function integrand(x, f, c) #return a tuple (real, complex) 
        #the code should handle real -> complex conversion
        f[1] = x[1]
        f[2] = x[1]^2 * 1im
    end
    res = integrate(integrand; dof=[[1,], [1,]], neval=totalstep, print=-1, type=ComplexF64, solver=alg, inplace=true, debug=true)
    config = res.config
    w = zeros(ComplexF64, 2)
    @inferred integrand(config.var[1], w, config) #make sure the type is inferred for the integrand function
    return res
end

# struct Weight <: AbstractVector
#     d::Tuple{Float64,Float64}
#     function Weight(a, b)
#         return new((a, b))
#     end
# end
# Base.zero(::Type{Weight}) = Weight(0.0, 0.0)
# Base.zero(::Weight) = Weight(0.0, 0.0)
# Base.abs(w::Weight) = w.d[1] + w.d[1]
# # Base.:^(w::Weight, i) = Weight(w.d^i, w.e^i)
# # Base.:*(w::Weight, c) = Weight(w.d * c, w.e * c)
# # Base.:/(w::Weight, c) = Weight(w.d / c, w.e / c)
# # Base.:+(a::Weight, b::Weight) = Weight(a.d + b.d, a.e * b.e)
# # Base.length(::Weight) = 1


# function Test_user_type(totalstep, alg)

#     function integrand(x, c) #return a tuple (real, complex) 
#         #the code should handle real -> complex conversion
#         return Weight(x[1], x[1]^2)
#     end
#     res = integrate(integrand; dof=[[1,],], neval=totalstep, print=-1, type=Weight, solver=alg, inplace=false, debug=true)
#     config = res.config
#     w = [Weight(0.0, 0.0),]
#     @inferred integrand(config.var[1], w, config) #make sure the type is inferred for the integrand function
#     return res
# end

@testset "Report" begin
    neval = 1000_00
    results = [
        Sphere1(neval, :vegas),
        Sphere2(neval, :vegas),
        TestComplex1(neval, :vegas),
        TestComplex2(neval, :vegas),
    ]
    for result in results
        @test redirect_stdout(devnull) do
            isnothing(report(result))
        end
    end
end

@testset "MCMC Sampler" begin
    neval = 1000_00
    println("MCMC tests")

    println("Constant with reweight goal")
    check(TestMCMCReweight(neval), 1)
    println("Sphere 2D")
    check(Sphere1(neval, :mcmc), π / 4.0)
    println("Sphere 2D + 3D")
    check(Sphere2(neval, :mcmc), [π / 4.0, 4.0 * π / 3.0 / 8])
    check(Sphere2(neval, :mcmc; offset=2), [π / 4.0, 4.0 * π / 3.0 / 8])
    check_vector(Sphere3(neval, :mcmc), [π / 4.0, [4.0 * π / 3.0 / 8, 4.0 * π / 3.0 / 4]])
    println("Discrete")
    check(TestDiscrete(neval, :mcmc), 6.0)
    println("Singular1")
    res = TestSingular1(neval, :mcmc)
    @time res = TestSingular1(neval, :mcmc)
    @time res = TestSingular1(neval * 2, :mcmc)
    println(res)
    # check(res, -4.0)
    # @test res.stdev[1] < 0.0004 #make there is no regression, vegas typically gives accuracy ~0.0002 with 1e5x10 evaluations
    println("Singular2")
    check(TestSingular2(neval, :mcmc), 1.3932)
    check(TestSingular2_CompositeVar(neval, :mcmc), 1.3932)
    check(TestSingular2_Continuous_HighDim(neval, :mcmc), 1.3932)

    neval = 1000_00
    println("Complex1")
    check_complex(TestComplex1(neval, :mcmc), 0.5 + 1.0 / 3 * 1im)
    println("Complex2")
    check_complex(TestComplex2(neval, :mcmc), [0.5, 1.0 / 3 * 1im])

end

@testset "Vegas Sampler" begin
    neval = 2000_00
    println("Vegas tests")

    println("Sphere 2D")
    check(Sphere1(neval, :vegas), π / 4.0)
    println("Sphere 2D + 3D")
    check(Sphere2(neval, :vegas), [π / 4.0, 4.0 * π / 3.0 / 8])
    check(Sphere2(neval, :vegas; offset=2), [π / 4.0, 4.0 * π / 3.0 / 8])
    check_vector(Sphere3(neval, :vegas), [π / 4.0, [4.0 * π / 3.0 / 8, 4.0 * π / 3.0 / 4]])
    println("Discrete")
    check(TestDiscrete(neval, :vegas), 6.0)
    println("Singular1")
    res = TestSingular1(neval, :vegas)
    @time res = TestSingular1(neval, :vegas)
    println(res)
    check(res, -4.0)
    @test res.stdev[1] < 0.0004 #make there is no regression, vegas typically gives accuracy ~0.0002 with 1e5x10 evaluations
    println("Singular2")
    check(TestSingular2(neval, :vegas), 1.3932)
    check(TestSingular2_CompositeVar(neval, :vegas), 1.3932)
    check(TestSingular2_Continuous_HighDim(neval, :vegas), 1.3932)

    neval = 2000_00
    println("Complex1")
    check_complex(TestComplex1(neval, :vegas), 0.5 + 1.0 / 3 * 1im)
    println("Complex2")
    check_complex(TestComplex2(neval, :vegas), [0.5, 1.0 / 3 * 1im])

    println("inplace Complex2")
    check_complex(TestComplex2_inplace(neval, :vegas), [0.5, 1.0 / 3 * 1im])

    # println("vector type")
    # check_vector(Test_user_type(neval, :vegas), [0.5, 1.0 / 3])
end

@testset "Markov-Chain Vegas" begin
    neval = 1000_00
    println("MC Vegas tests")

    # TODO: so far vegas MC doesn't work with Sphere1 and Sphere2. These integrals vanishes in some regimes, making the measurement of the normalization integral unreliable.
    println("Sphere1")
    check(Sphere1(neval, :vegasmc), π / 4.0)
    # check(Sphere2(neval), π / 4.0)
    println("Sphere2")
    res = Sphere2(neval, :vegasmc)
    println(res)
    check(res, [π / 4.0, 4.0 * π / 3.0 / 8])
    println("Sphere2 with offset")
    check(Sphere2(neval, :vegasmc; offset=2), [π / 4.0, 4.0 * π / 3.0 / 8])
    # check(Sphere3(neval), [π / 4.0, 4.0 * π / 3.0 / 8])
    check_vector(Sphere3(neval, :vegasmc), [π / 4.0, [4.0 * π / 3.0 / 8, 4.0 * π / 3.0 / 4]])
    println("Discrete")
    check(TestDiscrete(neval, :vegasmc), 6.0)
    println("Singular1")
    res = TestSingular1(neval, :vegasmc)
    @time res = TestSingular1(neval, :vegasmc)
    println(res)
    check(res, -4.0)
    @test res.stdev[1] < 0.0007 #make there is no regression, vegas typically gives accuracy ~0.0002 with 1e5x10 evaluations
    println("Singular2")
    check(TestSingular2(neval, :vegasmc), 1.3932)
    check(TestSingular2_CompositeVar(neval, :vegasmc), 1.3932)
    check(TestSingular2_Continuous_HighDim(neval, :vegasmc), 1.3932)

    @time TestSingular2_Continuous_HighDim(neval, :vegasmc)


    neval = 1000_00
    println("Complex1")
    check_complex(TestComplex1(neval, :vegasmc), 0.5 + 1.0 / 3 * 1im)
    println("Complex2")
    check_complex(TestComplex2(neval, :vegasmc), [0.5, 1.0 / 3 * 1im])

    println("inplace Complex2")
    check_complex(TestComplex2_inplace(neval, :vegasmc), [0.5, 1.0 / 3 * 1im])

    # println("vector type")
    # check_vector(Test_user_type(neval, :vegamcs), [0.5, 1.0 / 3])
end
