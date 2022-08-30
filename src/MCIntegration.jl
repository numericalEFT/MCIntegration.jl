module MCIntegration
using MPI
using Printf, Dates
using Random
using Graphs
using ProgressMeter
const RNG = Random.GLOBAL_RNG

include("utility/utility.jl")
using .MCUtility

include("distribution/distribution.jl")
using .Dist
export Dist
export FermiK
export Continuous, Discrete

include("statistics.jl")
export Result
include("configuration.jl")
export Configuration

include("mcmc/MCMC.jl")
using .MCMC
export MCMC

include("mc/MC.jl")
using .MC
export MC

include("main.jl")
export integrate
export report

end
