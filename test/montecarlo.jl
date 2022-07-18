function Sphere1(totalstep)
    function integrand(config)
        X = config.var[1]
        if (X[1]^2 + X[2]^2 < 1.0)
            return 1.0
        else
            return 0.0
        end
    end

    function measure(config)
        factor = 1.0 / config.reweight[config.curr]
        weight = integrand(config)
        config.observable += weight / abs(weight) * factor
    end

    T = MCIntegration.Tau(1.0, 1.0 / 2.0)
    dof = [[2,],] # number of T variable for the normalization and the integrand
    config = MCIntegration.Configuration((T,), dof, 0.0)
    result = MCIntegration.sample(config, integrand, measure; neval=totalstep, block=64, print=-1)
    # avg, err = MonteCarlo.sample(totalstep, (T,), dof, [0.0, ], integrand, measure; Nblock=64, print=-1)
    return result.mean, result.stdev
end

function Sphere2(totalstep)
    function integrand(config)
        X = config.var[1]
        if (X[1][1]^2 + X[1][2]^2 < 1.0)
            return 1.0
        else
            return 0.0
        end
    end

    function measure(config)
        factor = 1.0 / config.reweight[config.curr]
        weight = integrand(config)
        config.observable += weight / abs(weight) * factor
    end

    T = MCIntegration.TauPair(1.0, 1.0 / 2.0)
    dof = [[1,],] # number of T variable for the normalization and the integrand
    config = MCIntegration.Configuration((T,), dof, 0.0)
    result = MCIntegration.sample(config, integrand, measure; neval=totalstep, block=64, print=-1)
    # avg, err = MonteCarlo.sample(totalstep, (T,), dof, [0.0, ], integrand, measure; Nblock=64, print=-1)
    return result.mean, result.stdev
end

function Sphere3(totalstep; offset=0)
    function integrand(config)
        @assert config.curr == 1 || config.curr == 2
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
        factor = 1.0 / config.reweight[config.curr]
        weight = integrand(config)
        config.observable[config.curr] += weight / abs(weight) * factor
    end

    T = MCIntegration.Continuous(0.0, 1.0; offset=offset)
    dof = [[2,], [3,]] # number of T variable for the normalization and the integrand
    # config = MCIntegration.Configuration(totalstep, (T,), dof, [0.0, 0.0], neighbor=[[3, 2], [3, 1], [1, 2]])
    config = MCIntegration.Configuration((T,), dof, [0.0, 0.0], neighbor=[(1, 3), (1, 2)])
    result = MCIntegration.sample(config, integrand, measure; neval=totalstep, block=64, print=-1)
    # avg, err = MonteCarlo.sample(totalstep, (T,), dof, [0.0, ], integrand, measure; Nblock=64, print=-1)
    return result.mean, result.stdev
end

function Exponential1(totalstep)
    function integrand(config)
        X = config.var[1]
        return exp(-X[1])
    end

    function measure(config)
        factor = 1.0 / config.reweight[config.curr]
        weight = integrand(config)
        config.observable += weight / abs(weight) * factor
    end

    K = MCIntegration.RadialFermiK(1.0, 0.01)
    dof = [[1,],] # number of T variable for the normalization and the integrand
    config = MCIntegration.Configuration((K,), dof, 0.0)
    result = MCIntegration.sample(config, integrand, measure; neval=totalstep, block=64, print=-1)
    # avg, err = MonteCarlo.sample(totalstep, (T,), dof, [0.0, ], integrand, measure; Nblock=64, print=-1)
    return result.mean, result.stdev
end
const LorentzN = 1000

function Lorentz1(totalstep)
    function integrand(config)
        X = config.var[1]
        return 1.0 / ((X[1] - 1)^2 + (1.0 / LorentzN)^2)
    end

    function measure(config)
        factor = 1.0 / config.reweight[config.curr]
        weight = integrand(config)
        config.observable += weight / abs(weight) * factor
    end

    K = MCIntegration.RadialFermiK(1.0, 0.001)
    dof = [[1,],] # number of T variable for the normalization and the integrand
    config = MCIntegration.Configuration((K,), dof, 0.0)
    result = MCIntegration.sample(config, integrand, measure; neval=totalstep, block=64, print=-1)
    # avg, err = MonteCarlo.sample(totalstep, (T,), dof, [0.0, ], integrand, measure; Nblock=64, print=-1)
    return result.mean, result.stdev
end


function Exponential2(totalstep)
    function integrand(config)
        @assert config.curr == 1 || config.curr == 2
        X = config.var[1]
        if config.curr == 1
            return exp(-X[1])
        else
            return exp(-X[1] - X[2])
        end
    end

    function measure(config)
        factor = 1.0 / config.reweight[config.curr]
        weight = integrand(config)
        config.observable[config.curr] += weight / abs(weight) * factor
    end

    K = MCIntegration.RadialFermiK(1.0, 0.01)
    dof = [[1,], [2,]] # number of T variable for the normalization and the integrand
    config = MCIntegration.Configuration((K,), dof, [0.0, 0.0])
    result = MCIntegration.sample(config, integrand, measure; neval=totalstep, block=64, print=-1)
    # avg, err = MonteCarlo.sample(totalstep, (T,), dof, [0.0, ], integrand, measure; Nblock=64, print=-1)
    return result.mean, result.stdev
end


function Lorentz2(totalstep)
    function integrand(config)
        @assert config.curr == 1 || config.curr == 2
        X = config.var[1]
        if config.curr == 1
            return 1.0 / ((X[1] - 1)^2 + (1.0 / LorentzN)^2)
        else
            return 1.0 / ((X[1] - 1)^2 + (1.0 / LorentzN)^2) * exp(-X[2])
        end
    end

    function measure(config)
        factor = 1.0 / config.reweight[config.curr]
        weight = integrand(config)
        config.observable[config.curr] += weight / abs(weight) * factor
    end

    K = MCIntegration.RadialFermiK(1.0, 0.01)
    dof = [[1,], [2,]] # number of T variable for the normalization and the integrand
    config = MCIntegration.Configuration((K,), dof, [0.0, 0.0])
    result = MCIntegration.sample(config, integrand, measure; neval=totalstep, block=64, print=-1)
    # avg, err = MonteCarlo.sample(totalstep, (T,), dof, [0.0, ], integrand, measure; Nblock=64, print=-1)
    return result.mean, result.stdev
end

function TestDiscrete(totalstep)
    function integrand(config)
        x = config.var[1][1]
        return x
    end

    X = MCIntegration.Discrete(1, 3, adapt=true)
    dof = [[1,],] # number of X variable of the integrand
    config = MCIntegration.Configuration((X,), dof)

    result = MCIntegration.sample(config, integrand; neval=totalstep, niter=10, block=64, print=-1)
    # println(MCIntegration.summary(result))
    return result.mean, result.stdev
end

@testset "MonteCarlo Sampler" begin
    totalStep = 1000_000

    avg, err = Sphere1(totalStep)
    println("MC integration 1: $avg ± $err (exact: $(π / 4.0))")
    @test abs(avg - π / 4.0) < 5.0 * err
    # @test abs(avg[1] - π / 4.0) < 5.0 * err[1]

    avg, err = Sphere2(totalStep)
    println("MC integration 2: $avg ± $err (exact: $(π / 4.0))")
    @test abs(avg - π / 4.0) < 5.0 * err
    # @test abs(avg[1] - π / 4.0) < 5.0 * err[1]

    avg, err = Sphere3(totalStep)
    println("MC integration 3: $(avg[1]) ± $(err[1]) (exact: $(π / 4.0))")
    println("MC integration 3: $(avg[2]) ± $(err[2]) (exact: $(4.0 * π / 3.0 / 8))")
    @test abs(avg[1] - π / 4.0) < 5.0 * err[1]
    @test abs(avg[2] - π / 6.0) < 5.0 * err[2]

    avg, err = Sphere3(totalStep, offset=2)
    println("MC integration 3 with offset: $(avg[1]) ± $(err[1]) (exact: $(π / 4.0))")
    println("MC integration 3 with offset: $(avg[2]) ± $(err[2]) (exact: $(4.0 * π / 3.0 / 8))")
    @test abs(avg[1] - π / 4.0) < 5.0 * err[1]
    @test abs(avg[2] - π / 6.0) < 5.0 * err[2]

    avg, err = Exponential1(totalStep)
    println("MC integration 4: $avg ± $err (exact: $(1.0))")
    @test abs(avg - 1.0) < 5.0 * err

    avg, err = Exponential2(totalStep)
    println("MC integration 5: $(avg[1]) ± $(err[1]) (exact: $(1.0))")
    println("MC integration 5: $(avg[2]) ± $(err[2]) (exact: $(1.0))")
    @test abs(avg[1] - 1.0) < 5.0 * err[1]
    @test abs(avg[2] - 1.0) < 5.0 * err[2]

    avg, err = Lorentz1(totalStep)
    println("MC integration 6: $avg ± $err (exact: $((LorentzN / 2 * π + LorentzN * atan(LorentzN)))")
    @test abs(avg - (LorentzN / 2 * π + LorentzN * atan(LorentzN))) < 5.0 * err

    avg, err = Lorentz2(totalStep)
    println("MC integration 7: $(avg[1]) ± $(err[1]) (exact: $((LorentzN / 2 * π + LorentzN * atan(LorentzN))))")
    println("MC integration 7: $(avg[2]) ± $(err[2]) (exact: $((LorentzN / 2 * π + LorentzN * atan(LorentzN))))")
    @test abs(avg[1] - ((LorentzN / 2 * π + LorentzN * atan(LorentzN)))) < 5.0 * err[1]
    @test abs(avg[2] - ((LorentzN / 2 * π + LorentzN * atan(LorentzN)))) < 5.0 * err[2]

    avg, err = TestDiscrete(totalStep)
    println("MC integration 8: $avg ± $err (exact: 6.0)")
    @test abs(avg - 6.0) < 5.0 * err # the sum of possible values of X

end
