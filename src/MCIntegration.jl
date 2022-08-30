module MCIntegration
using MPI
using Printf, Dates
using Random
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

include("mcmc/MCMC.jl")
using .MCMC
export MCMC
export Configuration
export integrate, sample
export report

end
