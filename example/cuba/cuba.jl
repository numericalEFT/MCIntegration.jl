using Cuba
using MCIntegration
# using Plots
using PyCall
# using PythonCall
Vegas = pyimport("vegas")

function f1(x)
    return log(x[1]) / sqrt(x[1])
end

# x = LinRange(0.0, 1.0, 100)
# plot(x, f1.(x))

result = vegas((x, f) -> f[1] = f1(x[1]), maxevals=1e4)
@time result = vegas((x, f) -> f[1] = f1(x[1]), maxevals=1e4)

integ = Vegas.Integrator([[0, 1],])

result = integ(f1, nitn=10, neval=1e3)
@time result = integ(f1, nitn=10, neval=1e3)
println(result.summary())
println("result = $(result), Q = $(result.Q)")

function integrand(c)
    return f1(c.var[1][1])
end
result = integrate(c -> (f1(c.var[1][1])), neval=1e3, niter=10)
@time result = integrate(integrand, neval=1e3, niter=10, print=-1)

# X = Continuous(0.0, 1.0, grid=collect(LinRange(0.0, 1.0, 1000)), adapt=false)
# @time result = integrate(c -> (f1(c.var[1][1])), var=(X,), neval=1e6, niter=50, print=0)