using Cuba
using MCIntegration
using Plots
# using PythonCall

function f1(x)
    return log(x) / sqrt(x)
end
x = LinRange(0.0, 1.0, 100)
plot(x, f1.(x))


result = vegas((x, f) -> f[1] = f1(x[1]), maxevals=1e5)
result = integrate(c -> (f1(c.var[1][1])), neval=1e6)

@time result = vegas((x, f) -> f[1] = f1(x[1]), maxevals=1e5)
X = Continuous(0.0, 1.0, grid=collect(LinRange(0.0, 1.0, 1000)), adapt=false)
@time result = integrate(c -> (f1(c.var[1][1])), var=(X,), neval=1e6, niter=50, print=0)

