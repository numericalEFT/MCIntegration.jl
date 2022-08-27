"""
Benchmark three integrals
After 1e5 evaluations

Classical Vegas:
PyObject 0.24631(65)
PyObject 0.12316(33)
PyObject 0.06220(17)

Vegas + hypercube redistribution:
PyObject 0.24712(50)
PyObject 0.12354(25)
PyObject 0.06237(13)

Cuba:
1: 0.24681600683822702 ± 0.00029754211970684077 (prob.: 1.5893338049521866e-5)
2: 0.12341321349438042 ± 0.00014105471148060343 (prob.: 1.9926701015915427e-6)
3: 0.06232499578312799 ± 7.424773576055397e-5 (prob.: 7.508138821776811e-7)

MCIntegration currently fails
"""

using MCIntegration
using Cuba
using PyCall

Vegas = pyimport("vegas")

function f3(x)
    dx2 = 0.0
    for d in 1:4
        dx2 += (x[d] - 0.5)^2
    end
    f = exp(-200 * dx2) * 1000.0
    return [f, f * x[1], f * x[1]^2]
end

function f3cuba(x, out)
    dx2 = 0.0
    for d in 1:4
        dx2 += (x[d] - 0.5)^2
    end
    f = exp(-200 * dx2) * 1000.0
    out[1] = f
    out[2] = f * x[1]
    out[3] = f * x[1]^2
    return
end

integ = Vegas.Integrator([[0, 1], [0, 1], [0, 1], [0, 1]])
# adapt grid
# training = integ(f, nitn=10, neval=2000)
# final analysis
result = integ(f3, nitn=10, neval=1e4, beta=0.0)

integ2 = Vegas.Integrator([[0, 1], [0, 1], [0, 1], [0, 1]])
# adapt grid
# training = integ(f, nitn=10, neval=2000)
# final analysis
result = integ2(f3, nitn=10, neval=1e4)

result = vegas(f3cuba, 4, 3, maxevals=1e5)


result = integrate(neval=10000, dof=[[1], [1], [1], [1]], print=0) do c
    x = c.var[1]
    dx2 = 0.0
    for d in 1:4
        dx2 += (x[d] - 0.5)^2
    end
    f = exp(-200 * dx2) * 1000.0
    # f = exp(-1 * dx2)
    if c.curr == 1
        return f
    elseif c.curr == 2
        return f * (x[1] + 1e-4)
    else
        return f * (x[1] + 1e-4)^2
    end
end
MCIntegration.summary(result, [obs -> obs[1], obs -> obs[2], obs -> obs[3]]; verbose=-1)