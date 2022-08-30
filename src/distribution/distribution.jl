module Dist

abstract type AdaptiveMap end
abstract type Variable end
const MaxOrder = 16

include("sampler.jl")
include("variable.jl")
export Variable
export FermiK
export Continuous
export Discrete
end