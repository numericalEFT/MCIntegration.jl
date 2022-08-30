module MCMC

import ..Result
import ..report
import ..Configuration
import ..MPIreduceConfig!
import ..setweight!
import ..addConfig!
import ..clearStatistics!

using ..Dist
import ..Variable

using ..MCUtility

using Random, MPI
using LinearAlgebra
using Printf, Dates
using ProgressMeter

include("montecarlo.jl")
include("updates.jl")
export integrate, sample, report
end