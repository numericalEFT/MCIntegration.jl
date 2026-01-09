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

res = integrate((x, c) -> (f1(x)), neval=1e5, niter=10)
# @time result = integrate(integrand, neval=1e3, niter=10, print=-1)
println("MCIntegration.jl (Julia): ", res.mean[1], " ± ", res.stdev[1])


result = vegas((x, f) -> f[1] = f1(x), maxevals=1e6)
println("Cuba (C): ", result.integral[1], " ± ", result.error[1])

integ = Vegas.Integrator([[0, 1],])
result = integ(f1, nitn=10, neval=1e5, beta=0.0)
println("Classic Vegas (Python): ", result)

integ2 = Vegas.Integrator([[0, 1],])
result = integ2(f1, nitn=10, neval=1e5)
println("Vegas+ (Python): ", result)
# println(result.summary())
# println("result = $(result), Q = $(result.Q)")


# X = Continuous(0.0, 1.0, grid=collect(LinRange(0.0, 1.0, 1000)), adapt=false)
# @time result = integrate(c -> (f1(c.var[1][1])), var=(X,), neval=1e6, niter=50, print=0)