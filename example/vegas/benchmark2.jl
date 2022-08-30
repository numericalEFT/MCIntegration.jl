"""
Example from Cuba.jl documentation: https://giordano.github.io/Cuba.jl/stable/#One-dimensional-integral-1
Integral log(x)/sqrt(x) in [0, 1), expected to be -4
Evaluated 1e6 times

Cuba: -3.998162393712846 ± 0.00044066437168409556
Vegas classic:  -3.99798(14)
Vegas+hypercube redistribution: -3.999953(24)
Kristjan' Vegas: -3.999076429439272 +- 0.0004549401949589529
MCIntegration: -4.016640971329379 ± 0.018422223764188922

Both MCIntegration and Kristjan's vegas fail this example
"""

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

result = vegas((x, f) -> f[1] = f1(x[1]), maxevals=1e6)

integ = Vegas.Integrator([[0, 1],])
result = integ(f1, nitn=10, neval=1e5, beta=0.0)

integ2 = Vegas.Integrator([[0, 1],])
result = integ2(f1, nitn=10, neval=1e5)
println(result.summary())
println("result = $(result), Q = $(result.Q)")

function integrand(c)
    return f1(c.var[1][1])
end
result = integrate(c -> (f1(c.var[1][1])), neval=1e3, niter=10)
@time result = integrate(integrand, neval=1e3, niter=10, print=-1)

# X = Continuous(0.0, 1.0, grid=collect(LinRange(0.0, 1.0, 1000)), adapt=false)
# @time result = integrate(c -> (f1(c.var[1][1])), var=(X,), neval=1e6, niter=50, print=0)