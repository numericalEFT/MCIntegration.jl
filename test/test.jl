using MCIntegration
using JLD2
var = MCIntegration.Continuous(0.0, 1.0)
# var = MCIntegration.Tau(1.0)
config = MCIntegration.Configuration((var,), [[1,], [1,]])

function integrand(config)
    x = config.var[1][1]
    if config.curr == 1
        return x * 1.0
    else
        return x^2 * 1.0
    end
end

results = MCIntegration.sample(config, integrand; neval=1e4, block=64, niter=10, print=-1)
if isnothing(results) == false
    println(MCIntegration.summary(results, [obs -> obs[1], obs -> obs[2]]))
    # println(MCIntegration.summary(results, [obs -> obs[1],]))
end
# jldsave("test.jld", result=results)

