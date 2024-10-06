module MCIntegration
using Printf, Dates
using Random
using Graphs
using ProgressMeter
# using Measurements

const RNG = Random.GLOBAL_RNG
const TINY = eps(Float64(0)) * 1e50 # 4.940656458412466e-274
const EPSILON = eps(Float64)

abstract type ParallelBackend end
struct DefaultBackend <: ParallelBackend end
struct MPIBackend <: ParallelBackend end

function integrate(f, x, config::DefaultBackend)
end

integrate(f, x) = integrate(f, x, Defaultbackend())

# this is how vegas python package does it
# cdef double TINY = 10 ** (sys.float_info.min_10_exp + 50)  # smallest and biggest
# cdef double HUGE = 10 ** (sys.float_info.max_10_exp - 50)  # with extra headroom

include("utility/utility.jl")
using .MCUtility
export disable_threading

include("distribution/distribution.jl")
using .Dist
export Dist
export FermiK
export Continuous, Discrete, CompositeVar

include("statistics.jl")
export Result
include("configuration.jl")
export Configuration

include("main.jl")
export integrate
export report

include("vegas_mc/VegasMC.jl")
using .VegasMC
export VegasMC

include("vegas/Vegas.jl")
using .Vegas
export Vegas

include("mcmc/MCMC.jl")
using .MCMC
export MCMC


end
