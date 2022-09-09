using MCIntegration
# using JLD2
N = 8
alpha = 3.0
x1 = MCIntegration.Continuous(-1.0, 1.0; grid=collect(LinRange(-1.0, 1.0, N)), alpha=alpha)
x2 = MCIntegration.Continuous(0.0, 1.0; grid=collect(LinRange(0.0, 1.0, N)), alpha=alpha)
# x3, x4 = deepcopy(x2), deepcopy(x2)
# println("grid1: ", x1.grid)
# println("grid2: ", x2.grid)
# var = MCIntegration.Tau(1.0)
config = MCIntegration.Configuration(var=(x1, x2), dof=[[1, 3,],])

# function integrand(config)
#     x = config.var[1][1]
#     if config.curr == 1
#         return x * 1.0
#     else
#         return x^2 * 1.0
#     end
# end

println(config.var)

function integrand(X, config)
    x = [X[1][1], X[2][1], X[2][2], X[2][3]]
    dx2 = 0.0
    for d in 1:4
        dx2 += (x[d] - 0.5)^2
    end
    return exp(-dx2 * 100.0) * 1013.2118364296088
end

results = MCIntegration.integrate(integrand; config=config, block=16, niter=10, print=0)
# if isnothing(results) == false
#     # println(MCIntegration.summary(results, [obs -> obs[1], obs -> obs[2]]))
#     println(MCIntegration.summary(results, [obs -> obs[1],]))
#     # println(results.config.var[1].histogram)
#     # println("total: ", sum(results.config.var[1].histogram))
#     # dist = results.config.var[1].distribution
#     # for (gi, g) in enumerate(results.config.var[1].grid)
#     #     println(g, "   ", dist[gi])
#     # end
# end
# jldsave("test.jld", result=results)

