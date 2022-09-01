using MCIntegration
ENV["JULIA_PYTHONCALL_EXE"] = "/Users/kunchen/anaconda3/bin/python3"
using PythonCall
Vegas = pyimport("vegas")
np = pyimport("numpy")

function f1(c)
    X = c.var[1][1]
    return X
end
function f2(c)
    X = c.var[1][1]
    return X^2
end

X1 = Continuous(0.0, 1.0, alpha=3.0, grid=collect(LinRange(0.0, 1.0, 1024)), adapt=true)
res1 = integrate(f1;
    neval=1e5,
    var=(X,), dof=[[1,],], niter=10, print=0, solver=:MC)

grid1 = X1.grid
println("first 10 grid: ", grid[1:10])

X2 = Continuous(0.0, 1.0, alpha=3.0, grid=collect(LinRange(0.0, 1.0, 1024)), adapt=true)
res2 = integrate(f2;
    neval=1e5,
    var=(X2,), dof=[[1,],], niter=10, print=0, solver=:MC)

grid2 = X2.grid
println("first 10 grid: ", grid2[1:10])

config1 = deepcopy(res1.config)
config2 = deepcopy(res2.config)

# use grid1 to perform f2 integration

config1.curr = 1
res3 = integrate(f2; neval=1e7, dof=[[1,],], niter=10, print=0, solver=:MCMC, adapt=false, config=config1)

config1.curr = 2
res4 = integrate(f2; neval=1e7, dof=[[1,],], niter=10, print=0, solver=:MCMC, adapt=false, config=config1)

# now we let the solver adapt
res5 = integrate(f2; neval=1e7, dof=[[1,],], niter=10, print=0, solver=:MCMC, adapt=true)

# try a two integrals example
res6 = integrate(c -> [f1(c), f2(c)]; neval=1e7, dof=[[1,], [1,]], niter=10, print=0, solver=:MCMC, adapt=true)

# try a singular example to test the efficiency
# make sure it is correct
res7 = integrate(c -> log(c.var[1][1]) / sqrt(c.var[1][1]); neval=1e7, dof=[[1,],], niter=10, print=0, solver=:MCMC, adapt=true)
# make sure it is fast
res8 = integrate(c -> log(c.var[1][1]) / sqrt(c.var[1][1]); neval=1e5, dof=[[1,],], niter=10, print=0, solver=:MCMC, adapt=true)

res9 = integrate(c -> log(c.var[1][1]) / sqrt(c.var[1][1]); neval=1e5, niter=10, print=0, solver=:MC)
res8 = integrate(c -> log(c.var[1][1]) / sqrt(c.var[1][1]); neval=1e5, dof=[[1,],], niter=10, print=0, solver=:MCMC, adapt=true, config=res9.config)