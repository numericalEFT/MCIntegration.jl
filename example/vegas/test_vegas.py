import vegas
import numpy as np


def f(x):
    return np.log(x)/np.sqrt(x)


integ = vegas.Integrator([[0.0, 1.0], ])
re = integ(f, nitn=10, neval=1e5, alpha=1.5)
print(re)
print(re.summary())
