using MCIntegration
ENV["JULIA_PYTHONCALL_EXE"] = "/Users/kunchen/anaconda3/bin/python3"
using PythonCall
Vegas = pyimport("vegas")
np = pyimport("numpy")

function f1(x)
    return log(x[1]) / sqrt(x[1])
end
X = Continuous(0.0, 1.0, alpha=3.0, grid=collect(LinRange(0.0, 1.0, 1024)), adapt=true)
result = integrate(c -> log(c.var[1][1]) / sqrt(c.var[1][1]);
    neval=1e5,
    var=(X,), dof=[[1,],], niter=10, print=0, solver=:MCMC)

grid = X.grid
println("first 10 grid: ", grid[1:10])

N = 6250
y = np.random.random([N, 1])
x = np.zeros([N, 1])
jac = np.zeros(N)
m = Vegas.AdaptiveMap([grid,])
m.map(y, x, jac)
f = [f1(xx) * jac[xi-1] for (xi, xx) in enumerate(PyArray(x))]
println("f: ", f[1:10])
println(np.mean(f))
println(np.std(f) / sqrt(N))

avg = []
config = result.config
config.curr = 1
config.reweight = [0.2, 0.8]
Nblock = 16
# for i in 1:Nblock
mem = []
MCIntegration.clearStatistics!(config)
config = MCMC.markovchain_montecarlo(config, c -> f1(c.var[1][1]), N, 0, 0, []; measurefreq=1, mem=mem)
obs = config.observable / config.normalization
# println("Block $i: ", np.mean(mem), "+-", np.std(mem) / sqrt(length(mem)))
push!(avg, obs)
# end
println(avg)
println(np.mean(avg))
println(np.std(avg))
tseries1 = [curr == 1 ? weight : 0.0 for (curr, weight) in mem];
tseries2 = [curr == 2 ? weight : 0.0 for (curr, weight) in mem];
