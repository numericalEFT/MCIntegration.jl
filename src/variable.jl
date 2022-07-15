abstract type Variable end
abstract type Model end
const MaxOrder = 16


mutable struct FermiK{D} <: Variable
    # data::Vector{MVector{D,Float64}}
    data::Matrix{Float64}
    # data::Vector{Vector{Float64}}
    kF::Float64
    δk::Float64
    maxK::Float64
    offset::Int
    histogram::Vector{Float64}
    function FermiK(dim, kF, δk, maxK, size=MaxOrder; offset=0)
        @assert offset + 1 < size
        k = zeros(dim, size) .+ kF / sqrt(dim)
        # k0 = MVector{dim,Float64}([kF for i = 1:dim])
        # k0 = @SVector [kF for i = 1:dim]
        # k = [k0 for i = 1:size]
        return new{dim}(k, kF, δk, maxK, offset, [0.0,])
    end
end

Base.getindex(Var::FermiK{D}, i::Int) where {D} = Var.data[:, i]
function Base.setindex!(Var::FermiK{D}, v, i::Int) where {D}
    view(Var.data, :, i) .= v
end
Base.lastindex(Var::FermiK{D}) where {D} = size(Var.data)[2] # return index, not the value

mutable struct RadialFermiK <: Variable
    data::Vector{Float64}
    kF::Float64
    δk::Float64
    offset::Int
    histogram::Vector{Float64}
    function RadialFermiK(kF=1.0, δk=0.01, size=MaxOrder; offset=0)
        @assert offset + 1 < size
        k = [kF * (i - 0.5) / size for i = 1:size] #avoid duplication
        return new(k, kF, δk, offset, [0.0,])
    end
end

mutable struct BoseK{D} <: Variable
    data::Vector{SVector{D,Float64}}
    maxK::Float64
    histogram::Vector{Float64}
end

mutable struct Tau <: Variable
    data::Vector{Float64}
    λ::Float64
    β::Float64
    offset::Int
    histogram::Vector{Float64}
    function Tau(β=1.0, λ=0.5, size=MaxOrder; offset=0)
        @assert offset + 1 < size
        t = [β * (i - 0.5) / size for i = 1:size] #avoid duplication
        return new(t, λ, β, offset, [0.0,])
    end
end

mutable struct Continuous{G} <: Variable
    data::Vector{Float64}
    gidx::Vector{Int}
    lower::Float64
    range::Float64
    offset::Int
    grid::G
    width::Vector{Float64}
    histogram::Vector{Float64}
    accumulation::Vector{Float64}
    distribution::Vector{Float64}
    alpha::Float64
    function Continuous(lower::Float64, upper::Float64, size=MaxOrder; offset=0, grid::G=collect(LinRange(lower, upper, 129)), alpha=2.0) where {G}
        @assert offset + 1 < size
        @assert upper > lower + 2 * eps(1.0)
        t = LinRange(lower + eps(1.0), upper - eps(1.0), size) #avoid duplication
        gidx = [locate(grid, t[i]) for i = 1:size]
        # println(gidx)

        N = length(grid) - 1
        width = [grid[i+1] - grid[i] for i in 1:N]
        histogram = ones(N)
        # histogram = [1.0, 5.0, 1.0, 5.0]
        # distribution = histogram ./ width / sum(histogram)
        # accumulation = [sum(histogram[1:i]) / sum(histogram) for i in 1:N]
        # accumulation = [0.0, accumulation...] # start with 0.0 and end with 1.0
        # @assert (accumulation[1] ≈ 0.0) && (accumulation[end] ≈ 1.0)

        var = new{G}(t, gidx, lower, upper - lower, offset, grid, width, histogram, [], [], alpha)
        train!(var)
        return var
    end
end

function accumulate!(T::Continuous, idx::Int)
    T.histogram[T.gidx[idx]] += 1
end
# clearStatistics!(T::Continuous) = fill!(T.histogram, 1.0)
# addStatistics!(target::Continuous, income::Continuous) = (target.histogram .+= income.histogram)
function train!(T::Continuous)
    # distribution = T.histogram / sum(T.histogram)
    distribution = smooth(T.histogram, 6.0)
    distribution = rescale(distribution, T.alpha)
    distribution ./= sum(distribution)
    accumulation = [sum(distribution[1:i]) for i in 1:length(distribution)]
    T.accumulation = [0.0, accumulation...] # start with 0.0 and end with 1.0
    T.distribution = distribution ./ T.width
    @assert (T.accumulation[1] ≈ 0.0) && (T.accumulation[end] ≈ 1.0) "$(T.accumulation)"
end

mutable struct Angle <: Variable
    data::Vector{Float64}
    λ::Float64
    offset::Int
    histogram::Vector{Float64}
    function Angle(λ=0.5, size=MaxOrder; offset=0)
        @assert offset + 1 < size
        theta = [π * (i - 0.5) / size for i = 1:size] #avoid dulication
        return new(theta, λ, offset, [0.0,])
    end
end


mutable struct TauPair <: Variable
    data::Vector{MVector{2,Float64}}
    λ::Float64
    β::Float64
    offset::Int
    histogram::Vector{Float64}
    function TauPair(β=1.0, λ=0.5, size=MaxOrder; offset=0)
        @assert offset + 1 < size
        t = [@MVector [β * (i - 0.4) / size, β * (i - 0.6) / size] for i = 1:size] #avoid duplication
        return new(t, λ, β, offset, [0.0,])
    end
end

mutable struct Discrete <: Variable
    data::Vector{Int}
    lower::Int
    upper::Int
    size::Int
    offset::Int
    accumulation::Vector{Float64}
    histogram::Vector{Float64}
    function Discrete(bound::Union{Tuple{Int,Int},Vector{Int}}, size=MaxOrder; offset=0)
        return Discrete([bound[0], bound[1]], size; offset=offset)
    end
    function Discrete(lower::Int, upper::Int, size=MaxOrder; offset=0)
        d = [i for i = 1:size] #avoid dulication
        @assert offset + 1 < size
        @assert upper > lower
        histogram = ones(upper - lower + 1) / (upper - lower + 1)
        accumulation = [sum(histogram[1:i]) for i = 1:upper-lower+1]
        return new(d, lower, upper, upper - lower + 1, offset, accumulation, histogram)
    end
end

# mutable struct ContinuousND{D} <: Variable
#     data::Vector{Float64}
#     lower::Vector{Float64}
#     range::Vector{Float64}
#     offset::Int
#     function ContinuousND{dim}(lower, upper, size=MaxOrder; offset=0) where {dim}
#         if lower isa Number
#             lower = ones(Float64, dim) * lower
#         else
#             @assert length(lower) == dim && eltype(lower) isa Number
#         end
#         if upper isa Number
#             upper = ones(Float64, dim) * upper
#         else
#             @assert length(upper) == dim && eltype(upper) isa Number
#         end
#         @assert offset + 1 < size
#         @assert all(x -> x > 0, upper .- lower)
#         println(lower, ", ", upper)

#         ######## deterministic initialization #####################
#         t = []
#         for i in 1:size
#             for d in 1:dim
#                 # the same value should not appear twice!
#                 init = lower[d] + (upper[d] - lower[d]) * ((i - 1) * dim + d - 0.5) / (size * dim)
#                 @assert lower[d] <= init <= upper[d]
#                 append!(t, init)
#             end
#         end

#         return new{dim}(t, lower, upper .- lower, offset)
#     end
# end

accumulate!(var::Variable, idx) = nothing
# clearStatistics!(Var::Variable) = ing
train!(Var::Variable) = nothing
# addStatistics!(target::Variable, income::Variable) = nothing
clearStatistics!(T::Variable) = fill!(T.histogram, 1.0)
addStatistics!(target::Variable, income::Variable) = (target.histogram .+= income.histogram)

Base.getindex(Var::Variable, i::Int) = Var.data[i]
function Base.setindex!(Var::Variable, v, i::Int)
    Var.data[i] = v
end
Base.firstindex(Var::Variable) = 1 # return index, not the value
Base.lastindex(Var::Variable) = length(Var.data) # return index, not the value



# struct Uniform{T,D} <: Model
#     lower::T
#     upper::T
#     function Uniform{T,D}(lower, upper) where {T<:Number,D}
#         return new{T,D}(lower, upper)
#     end
# end

# mutable struct Var{T,D,M} <: Variable
#     data::T
#     model::M
#     offset::Int
#     function Var(model{type,D}::Model, size=MaxOrder; offset=0) where {type<:Number,D}
#         # lower, upper = model.lower, model.upper
#         # k = zeros(type, dim, size + offset)
#         # for i in 1:size
#         #     for d in 1:dim
#         #         init = lower[d] + (upper[d] - lower[d]) * ((i - 1) * dim + d - 0.5) / (size * dim)
#         #         @assert lower[d] <= init <= upper[d]
#         #         k[d, i+offset] = init
#         #     end
#         # end
#         if D == 1
#             data = zeros(type, size)
#         else
#             data = zeros(type, (D, size))
#         end
#         return new{typeof(data),D,typeof(model)}(data, model, offset)
#     end
# end

# Base.getindex(var::Var{T,1,M}, i::Int) where {T,M} = var.data[i]
# function Base.setindex!(var::Var{T,1,M}, v, i::Int) where {T,M}
#     var.data[i] = v
# end
# Base.lastindex(var::Var{T,1,M}) where {T,M} = length(var.data) # return index, not the value

# Base.getindex(var::Var{T,D,M}, i::Int) where {T,D,M} = var.data[:, i]
# function Base.setindex!(var::Var{T,D,M}, v, i::Int) where {T,M}
#     var.data[:, i] = v
# end
# Base.lastindex(var::Var{T,D,M}) where {T,D,M} = size(var.data)[2] # return index, not the value
