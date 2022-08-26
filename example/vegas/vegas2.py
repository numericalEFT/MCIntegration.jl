import vegas
import math
import gvar as gv


def f(x):
    dx2 = 0.0
    for d in range(4):
        dx2 += (x[d] - 0.5) ** 2
    f = math.exp(-200 * dx2)
    return [f, f * x[0], f * x[0] ** 2]


integ = vegas.Integrator(4 * [[0, 1]])

# adapt grid
training = integ(f, nitn=10, neval=2000)

# final analysis
result = integ(f, nitn=10, neval=10000)
print('I[0] =', result[0], '  I[1] =', result[1], '  I[2] =', result[2])
print('Q = %.2f\n' % result.Q)
print('<x> =', result[1] / result[0])
print(
    'sigma_x**2 = <x**2> - <x>**2 =',
    result[2] / result[0] - (result[1] / result[0]) ** 2
)
print('\ncorrelation matrix:\n', gv.evalcorr(result))
