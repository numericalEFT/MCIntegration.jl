using MCIntegration
var = MCIntegration.Continuous([0.0, 1.0], 1.0 / 2.0)
config = MCIntegration.Configuration((var,), [[1,],])

function integrand(config)
    x = config.var[1][1]
    if config.curr == 1
        return x * 0.0001
    else
        return x^2 * 100
    end
end

results = MCIntegration.sample(config, integrand; neval=1e4, block=64, niter=10, print=-1)
println(results)

