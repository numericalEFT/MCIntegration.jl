module MCIntegration
using MPI
using Printf, Dates
using Random
using Graphs
using ProgressMeter

const RNG = Random.GLOBAL_RNG
const TINY = 1e-10

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

include("vegas_mc/VegasMC.jl")
using .VegasMC
export VegasMC

include("vegas/Vegas.jl")
using .Vegas
export Vegas

include("main.jl")
export integrate
export report

end
