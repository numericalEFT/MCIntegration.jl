"""
Benchmark 1/(1-cos(x)*cos(y)*cos(z)) with different codes.
Exact: 1.3932

After 2e6 total evaluations
Kristjan's code: 1.391124528441497 +- 0.0011584630917009374
classic Vegas: 1.39114(48)
Vegas + hypercube redistribution: 1.39314(15)
Cuba: 1.3922290364427665 ± 0.0010906210048935345
"""
using MCIntegration
using PyCall
using Cuba
Vegas = pyimport("vegas")

function f(x)
    return 1.0 / (1.0 - cos(x[1]) * cos(x[2]) * cos(x[3])) / π^3
end

function fc(x)
    return 1.0 / (1.0 - cos(x[1] * π) * cos(x[2] * π) * cos(x[3] * π))
end

# MCIntegration : 1.3961243672238552 ± 0.008180219118720399
integrate(neval=200000, var=(Continuous(0.0, 1π, alpha=3.0, adapt=true),), dof=[[3,],]) do config
    x = config.var[1]
    return f(x)
end

# classic Vegas : 1.39114(48)
integ = Vegas.Integrator([[0, 1π], [0, 1π], [0, 1π]])
result = integ(f, nitn=10, neval=2e5, beta=0.0, alpha=0.5)
println(result)

# Vegas plus (vegas + hypercube) : 1.39314(15)
integ2 = Vegas.Integrator([[0, 1π], [0, 1π], [0, 1π]])
result = integ2(f, nitn=10, neval=2e5, alpha=0.5)
println(result)

# Cuba: 1.3922290364427665 ± 0.0010906210048935345
result = vegas((x, g) -> g[1] = fc(x), 3, maxevals=2e6)
@time result = vegas((x, g) -> g[1] = fc(x), 3, maxevals=2e6)
println(result)