module Dist
using StaticArrays
using LinearAlgebra
import ..TINY

abstract type AdaptiveMap end

"""
    abstract type Variable end

Abstract Type of all variable pools. 
"""
abstract type Variable end

const MaxOrder = 16

include("common.jl")
include("variable.jl")
include("sampler.jl")
export Variable
export FermiK
export Continuous
export Discrete
export CompositeVar
export create!, shift!, swap!
export createRollback!, shiftRollback!, swapRollback!
end