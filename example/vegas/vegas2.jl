using MCIntegration

result = integrate(neval=10000, dof=[[1], [1], [1], [1]], print=0) do c
    X = c.var[1]
    dx2 = 0.0
    for d in 1:4
        dx2 += (X[d] - 0.5)^2
    end
    f = exp(-200 * dx2)
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