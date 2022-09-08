module VegasMC

import ..Result
import ..report
import ..Configuration
import ..MPIreduceConfig!
import ..addConfig!
import ..clearStatistics!

import ..TINY

using ..Dist
import ..Variable

using ..MCUtility

using Random, MPI
using LinearAlgebra
using Printf, Dates

include("montecarlo.jl")
include("updates.jl")
end