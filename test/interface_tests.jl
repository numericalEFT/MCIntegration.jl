using MCIntegration

integrand = (x, c) -> [1.0]
vars = Continuous(0, 1)
dof = [(1,)]
integrate(integrand; dof, vars)
