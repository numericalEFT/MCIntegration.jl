abstract type Variable end
const MaxOrder = 16

mutable struct FermiK{D} <: Variable
    data::Vector{MVector{D,Float64}}
    # data::Vector{Vector{Float64}}
    kF::Float64
    δk::Float64
    maxK::Float64
    offset::Int
    function FermiK(dim, kF, δk, maxK, size = MaxOrder; offset = 0)
        @assert offset + 1 < size
        k0 = MVector{dim,Float64}([kF for i = 1:dim])
        # k0 = @SVector [kF for i = 1:dim]
        k = [k0 for i = 1:size]
        return new{dim}(k, kF, δk, maxK, offset)
    end
end

mutable struct RadialFermiK <: Variable
    data::Vector{Float64}
    kF::Float64
    δk::Float64
    offset::Int
    function RadialFermiK(kF = 1.0, δk = 0.01, size = MaxOrder; offset = 0)
        @assert offset + 1 < size
        k = [kF * (i - 0.5) / size for i = 1:size] #avoid duplication
        return new(k, kF, δk, offset)
    end
end

mutable struct BoseK{D} <: Variable
    data::Vector{SVector{D,Float64}}
    maxK::Float64
end

mutable struct Tau <: Variable
    data::Vector{Float64}
    λ::Float64
    β::Float64
    offset::Int
    function Tau(β = 1.0, λ = 0.5, size = MaxOrder; offset = 0)
        @assert offset + 1 < size
        t = [β * (i - 0.5) / size for i = 1:size] #avoid duplication
        return new(t, λ, β, offset)
    end
end

mutable struct Continuous <: Variable
    data::Vector{Float64}
    λ::Float64
    lower::Float64
    range::Float64
    offset::Int
    function Continuous(bound, λ = nothing, size = MaxOrder; offset = 0)
        lower, upper = bound
        @assert offset + 1 < size
        @assert upper > lower
        @assert isnothing(λ) || (0 < λ < (upper - lower))
        t = [lower + (upper - lower) * (i - 0.5) / size for i = 1:size] #avoid duplication

        if isnothing(λ)
            λ = (upper - lower) / 2.0
        end

        return new(t, λ, lower, upper - lower, offset)
    end
end

mutable struct Angle <: Variable
    data::Vector{Float64}
    λ::Float64
    offset::Int
    function Angle(λ = 0.5, size = MaxOrder; offset = 0)
        @assert offset + 1 < size
        theta = [π * (i - 0.5) / size for i = 1:size] #avoid dulication
        return new(theta, λ, offset)
    end
end


mutable struct TauPair <: Variable
    data::Vector{MVector{2,Float64}}
    λ::Float64
    β::Float64
    offset::Int
    function TauPair(β = 1.0, λ = 0.5, size = MaxOrder; offset = 0)
        @assert offset + 1 < size
        t = [@MVector [β * (i - 0.4) / size, β * (i - 0.6) / size] for i = 1:size] #avoid duplication
        return new(t, λ, β, offset)
    end
end

mutable struct Discrete <: Variable
    data::Vector{Int}
    lower::Int
    upper::Int
    size::Int
    offset::Int
    function Discrete(lower, upper, size = MaxOrder; offset = 0)
        d = [i for i = 1:size] #avoid dulication
        @assert offset + 1 < size
        @assert upper > lower
        return new(d, lower, upper, upper - lower + 1, offset)
    end
end


Base.getindex(Var::Variable, i::Int) = Var.data[i]
function Base.setindex!(Var::Variable, v, i::Int)
    Var.data[i] = v
end
Base.firstindex(Var::Variable) = 1 # return index, not the value
Base.lastindex(Var::Variable) = length(Var.data) # return index, not the value
