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

Base.getindex(Var::FermiK{D}, i::Int) where {D} = view(Var.data, :, i)
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

### variables that uses a vegas+ algorithm for impotrant sampling ###
# mutable struct Vegas{D,G} <: Variable
#     permutation::Vector{Int}
#     uniform::Matrix{Float64}
#     data::Matrix{Float64}
#     gidx::Vector{Int}
#     offset::Int
#     grid::G

#     width::Vector{Float64}
#     histogram::Vector{Float64}
#     accumulation::Vector{Float64}
#     distribution::Vector{Float64}

#     alpha::Float64
#     beta::Float64
#     adapt::Bool
# end

mutable struct Continuous{G} <: Variable
    data::Vector{Float64}
    gidx::Vector{Int}
    lower::Float64
    range::Float64
    offset::Int
    grid::G
    width::Vector{Float64}
    histogram::Vector{Float64} # length(grid) - 1
    accumulation::Vector{Float64} # length(grid)
    distribution::Vector{Float64} # length(grid) - 1
    alpha::Float64
    adapt::Bool
    function Continuous(lower::Float64, upper::Float64, size=MaxOrder; offset=0, grid::G=collect(LinRange(lower, upper, 1000)), alpha=2.0, adapt=true) where {G}
        @assert offset + 1 < size
        size = size + 1 # need one more element as cache for the swap operation
        @assert upper > lower + 2 * eps(1.0)
        t = LinRange(lower + (upper - lower) / size, upper - (upper - lower) / size, size) #avoid duplication
        gidx = [locate(grid, t[i]) for i = 1:size]
        # println(gidx)

        N = length(grid) - 1
        width = [grid[i+1] - grid[i] for i in 1:N]
        histogram = ones(N)

        var = new{G}(t, gidx, lower, upper - lower, offset, grid, width, histogram, [], [], alpha, adapt)

        train!(var)

        return var
    end
end

function Base.show(io::IO, var::Continuous)
    print(io, (var.adapt ? "Adaptive" : "Nonadaptive") * " continuous variable in the domain [$(var.lower), $(var.lower+var.range))."
              * (" Max variable number = $(length(var.data)-1-var.offset).")
              * (var.adapt ? " Learning rate = $(var.alpha)." : "")
              * (var.offset > 0 ? " Offset = $(var.offset)." : "")
    )
end

function accumulate!(T::Continuous, idx::Int)
    if T.adapt
        T.histogram[T.gidx[idx]] += 1
    end
end
function train_bk!(T::Continuous)
    distribution = smooth(T.histogram, 6.0)
    distribution = rescale(distribution, T.alpha)
    distribution ./= sum(distribution)
    accumulation = [sum(distribution[1:i]) for i in 1:length(distribution)]
    T.accumulation = [0.0, accumulation...] # start with 0.0 and end with 1.0
    T.distribution = distribution ./ T.width
    # println(T.distribution)
    # println(T.accumulation)
    @assert (T.accumulation[1] ≈ 0.0) && (T.accumulation[end] ≈ 1.0) "$(T.accumulation)"
end
"""
Vegas adaptive map
"""
function train!(T::Continuous)
    distribution = smooth(T.histogram, 6.0)
    distribution = rescale(distribution, T.alpha)
    newgrid = similar(T.grid)
    newgrid[1] = T.grid[1]
    newgrid[end] = T.grid[end]

    # nbins = length(T.grid) - 1
    # avgperbin = sum(distribution) / nbins
    # # **** redefine the size of each bin  ****
    # newgrid = zeros(nbins)
    # thisbin = 0.0
    # ibin = 0
    # # we are trying to determine
    # #   Int[ f(g) dg, {g, g_{i-1},g_i}] == I/N_g
    # #   where I == avgperbin
    # for newbin in 2:nbins  # all but the last bin, which is 1.0
    #     while thisbin < avgperbin
    #         ibin += 1
    #         thisbin += distribution[ibin]
    #         # prev = cur
    #         # cur = T.grid[ibin]
    #         # Explanation is in order :
    #         #   prev    -- g^{old}_{l-1}
    #         #   cur     -- g^{old}_l
    #         #   thisbin -- Sm = f_{l-k}+.... +f_{l-2}+f_{l-1}+f_l
    #         #   we know that  Sm is just a bit more than we need, i.e., I/N_g, hence we need to compute how much more
    #         #   using linear interpolation :
    #         #   g^{new} = g_l - (g_l-g_{l-1}) * (f_{l-k}+....+f_{l-2}+f_{l-1}+f_l - I/N_g)/f_l
    #         #    clearly
    #         #         if I/N_g == f_{l-k}+....+f_{l-2}+f_{l-1}+f_l
    #         #            we will get g^{new} = g_l
    #         #     and if I/N_g == f_{l-k}+....+f_{l-2}+f_{l-1}
    #         #            we will get g^{new} = g_{l-1}
    #         #     and if I/N_g  is between the two possibilities, we will get linear interpolation between
    #         #     g_{l-1} and g_l
    #         #
    #         # thisbin <- (f_{l-k}+....+f_{l-2}+f_{l-1}+f_l - I/N_g)
    #     end
    #     thisbin -= avgperbin
    #     # delta <-  (g_l-g_{l-1})*(f_{l-k}+....+f_{l-2}+f_{l-1}+f_l - I/N_g)
    #     delta = (T.grid[ibin+1] - T.grid[ibin]) * thisbin
    #     # cur is the closest point from the old mesh, while delta/imp is the correction using linear interpolation.
    #     newgrid[newbin] = T.grid[ibin+1] - delta / distribution[ibin]
    # end


    # See the paper https://arxiv.org/pdf/2009.05112.pdf Eq.(20)-(22).
    j = 0         # self_x index
    acc_f = 0.0   # sum(avg_f) accumulated
    avg_f = distribution
    # println(distribution)
    # amount of acc_f per new increment
    # the Eq.(20) in the original paper use length(T.grid) as the denominator. It is not correct.
    f_ninc = sum(avg_f) / (length(T.grid) - 1)
    for i in 2:length(T.grid)-1
        # println("working on i= $i")
        while acc_f < f_ninc
            j += 1
            acc_f += avg_f[j]
            # println("inner j=$j, acc_f=$acc_f")
        end
        acc_f -= f_ninc
        # println("outer j=$j, acc_f=$acc_f")
        # println(j, ", ", length(T.grid))
        newgrid[i] = T.grid[j+1] - (acc_f / avg_f[j]) * (T.grid[j+1] - T.grid[j])
    end
    newgrid[end] = T.grid[end] # make sure the last element is the same as the last element of the original grid
    T.grid = newgrid

    T.distribution = zeros(length(T.grid) - 1)
    T.accumulation = zeros(length(T.grid))
    for i in eachindex(T.distribution)
        T.distribution[i] = 1.0
        T.accumulation[i] = T.grid[i] - T.grid[1]
    end
    T.accumulation[end] = 1.0
    # println(newgrid)
    # println(T.distribution)
    # println(T.accumulation)
    @assert (T.accumulation[1] ≈ 0.0) && (T.accumulation[end] ≈ 1.0) "$(T.accumulation)"
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
    histogram::Vector{Float64}
    accumulation::Vector{Float64}
    distribution::Vector{Float64}
    alpha::Float64
    adapt::Bool
    function Discrete(bound::Union{Tuple{Int,Int},Vector{Int}}, size=MaxOrder; offset=0, alpha=2.0, adapt=true)
        return Discrete([bound[0], bound[1]], size; offset=offset, alpha=alpha, adapt=adapt)
    end
    function Discrete(lower::Int, upper::Int, size=MaxOrder; offset=0, alpha=2.0, adapt=true)
        @assert offset + 1 < size
        size = size + 1 # need one more element as cache for the swap operation
        d = collect(Iterators.take(Iterators.cycle(lower:upper), size)) #avoid dulication
        @assert upper >= lower
        histogram = ones(upper - lower + 1)
        newVar = new(d, lower, upper, upper - lower + 1, offset, histogram, [], [], alpha, adapt)
        train!(newVar)
        return newVar
    end
end

function Base.show(io::IO, var::Discrete)
    print(io, (var.adapt ? "Adaptive" : "Nonadaptive") * " discrete variable in the domain [$(var.lower), ..., $(var.upper)]."
              * (" Max variable number = $(length(var.data)-1-var.offset).")
              * (var.adapt ? " Learning rate = $(var.alpha)." : "")
              * (var.offset > 0 ? " Offset = $(var.offset)." : "")
    )
end

function accumulate!(T::Discrete, idx::Int)
    if T.adapt
        gidx = T[idx] - T.lower + 1
        T.histogram[gidx] += 1
    end
end
function train!(T::Discrete)
    # return
    # distribution = smooth(T.histogram, 6.0)
    distribution = deepcopy(T.histogram)
    distribution = rescale(distribution, T.alpha)
    distribution ./= sum(distribution)
    accumulation = [sum(distribution[1:i]) for i in 1:length(distribution)]
    T.accumulation = [0.0, accumulation...] # start with 0.0 and end with 1.0
    T.distribution = distribution
    @assert (T.accumulation[1] ≈ 0.0) && (T.accumulation[end] ≈ 1.0) "$(T.accumulation)"
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
clearStatistics!(T::Variable) = fill!(T.histogram, 1.0e-10)
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
