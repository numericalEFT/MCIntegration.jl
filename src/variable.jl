abstract type Variable end
abstract type Model end
const MaxOrder = 16

"""
mutable struct Configuration

    Struct that contains everything needed for MC.

    There are three different componenets:

 # Members

 ## Static parameters

 - `seed`: seed to initialize random numebr generator, also serves as the unique pid of the configuration

 - `rng`: a MersenneTwister random number generator, seeded by `seed`

 - `para`: user-defined parameter, set to nothing if not needed

 - `totalStep`: the total number of updates for this configuration

 - `var`: TUPLE of variables, each variable should be derived from the abstract type Variable, see variable.jl for details). Use a tuple rather than a vector improves the performance.

 ## integrand properties
 
- `neighbor::Vector{Tuple{Int, Int}}` : vector of tuples that defines the neighboring integrands. Two neighboring integrands are directly connected in the Markov chain. 
    e.g., [(1, 2), (2, 3)] means the integrand 1 and 2 are neighbor, and 2 and 3 are neighbor.  
   The neighbor vector defines a undirected graph showing how the integrands are connected. Please make sure all integrands are connected.
   By default, we assume the N integrands are in the increase order, meaning the neighbor will be set to [(N+1, 1), (1, 2), (2, 4), ..., (N-1, N)], where the first N entries are for diagram 1, 2, ..., N and the last entry is for the normalization diagram. Only the first diagram is connected to the normalization diagram.
   Only highly correlated integrands are not highly correlated should be defined as neighbors. Otherwise, most of the updates between the neighboring integrands will be rejected and wasted.

 - `dof::Vector{Vector{Int}}`: degrees of freedom of each integrand, e.g., [[0, 1], [2, 3]] means the first integrand has zero var#1 and one var#2; while the second integrand has two var#1 and 3 var#2. 

 - `observable`: observables that is required to calculate the integrands, will be used in the `measure` function call.
    It is either an array of any type with the common operations like +-*/^ defined. 

 - `reweight`: reweight factors for each integrands. The reweight factor of the normalization diagram is assumed to be 1. Note that you don't need to explicitly add the normalization diagram. 

 - `visited`: how many times this integrand is visited by the Markov chain.

 ## current MC state

 - `step`: the number of MC updates performed up to now

 - `curr`: the current integrand, initialize with 1

 - `norm`: the index of the normalization diagram. `norm` is larger than the index of any user-defined integrands 

 - `normalization`: the accumulated normalization factor. Physical observable = Configuration.observable/Configuration.normalization.

 - `absWeight`: the abolute weight of the current integrand. User is responsible to initialize it after the contructor is called.

 - `propose/accept`: array to store the proposed and accepted updates for each integrands and variables.
    Their shapes are (number of updates X integrand number X max(integrand number, variable number).
    The last index will waste some memory, but the dimension is small anyway.
"""
mutable struct Configuration{V,P,O}
    ########### static parameters ###################
    seed::Int # seed to initialize random numebr generator, also serves as the unique pid of the configuration
    rng::MersenneTwister # random number generator seeded by seed
    para::P
    totalStep::Int64
    var::V

    ########### integrand properties ##############
    neighbor::Vector{Vector{Int}}
    dof::Vector{Vector{Int}} # degrees of freedom
    observable::O  # observables for each integrand
    reweight::Vector{Float64}
    visited::Vector{Float64}

    ############# current state ######################
    step::Int64
    curr::Int # index of current integrand
    norm::Int # index of the normalization diagram
    normalization::Float64 # normalization factor for observables
    absWeight::Float64 # the absweight of the current diagrams. Store it for fast updates

    propose::Array{Float64,3} # updates index, integrand index, integrand index
    accept::Array{Float64,3} # updates index, integrand index, integrand index 

    """
    Configuration(totalStep, var::V, dof, obs::O; para::P=nothing, state=nothing, reweight=nothing, seed=nothing, neighbor=Vector{Vector{Int}}([])) where {V,P,O}

    Create a Configuration struct

 # Arguments

 ## Static parameters

 - `totalStep`: the total number MC steps of each block (one block, one configuration)

 - `var`: TUPLE of variables, each variable should be derived from the abstract type Variable, see variable.jl for details). Use a tuple rather than a vector improves the performance.

 - `dof::Vector{Vector{Int}}`: degrees of freedom of each integrand, e.g., [[0, 1], [2, 3]] means the first integrand has zero var#1 and one var#2; while the second integrand has two var#1 and 3 var#2. 

 - `obs`: observables that is required to calculate the integrands, will be used in the `measure` function call
    It is either an array of any type with the common operations like +-*/^ defined. 

 - `para`: user-defined parameter, set to nothing if not needed

 - `reweight`: reweight factors for each integrands. If not set, then all factors will be initialized with one.

 - `seed`: seed to initialize random numebr generator, also serves as the unique pid of the configuration. If it is nothing, then use RandomDevice() to generate a random seed in [1, 1000_1000]

- `neighbor::Vector{Tuple{Int, Int}}` : vector of tuples that defines the neighboring integrands. Two neighboring integrands are directly connected in the Markov chain. 
    e.g., [(1, 2), (2, 3)] means the integrand 1 and 2 are neighbor, and 2 and 3 are neighbor.  
   The neighbor vector defines a undirected graph showing how the integrands are connected. Please make sure all integrands are connected.
   By default, we assume the N integrands are in the increase order, meaning the neighbor will be set to [(N+1, 1), (1, 2), (2, 4), ..., (N-1, N)], where the first N entries are for diagram 1, 2, ..., N and the last entry is for the normalization diagram. Only the first diagram is connected to the normalization diagram.
   Only highly correlated integrands are not highly correlated should be defined as neighbors. Otherwise, most of the updates between the neighboring integrands will be rejected and wasted.
"""
    function Configuration(totalStep, var::V, dof, obs::O; para::P=nothing, reweight=nothing, seed=nothing, neighbor::Union{Vector{Vector{Int}},Vector{Tuple{Int,Int}},Nothing}=nothing) where {V,P,O}
        @assert totalStep > 0 "Total step should be positive!"
        # @assert O <: AbstractArray "observable is expected to be an array. Noe get $(typeof(obs))."
        @assert V <: Tuple{Vararg{Variable}} || V <: Tuple{Variable} "Configuration.var must be a tuple of Variable to maximize efficiency. Now get $(typeof(V))"
        Nv = length(var) # number of variables

        ################# integrand initialization #########################
        @assert typeof(dof) == Vector{Vector{Int}} "Configuration.dof should be with a type of Vector{Vector{Int}} to avoid mistakes. Now get $(typeof(dof))"
        # add normalization diagram to dof
        dof = deepcopy(dof) # don't modify the input dof
        push!(dof, zeros(Int, length(var))) # add the degrees of freedom for the normalization diagram

        Nd = length(dof) # number of integrands + renormalization diagram
        @assert Nd > 1 "At least one integrand is required."
        # make sure dof has the correct size that matches var and neighbor
        for nv in dof
            @assert length(nv) == Nv "Each element of `dof` should have the same dimension as `var`"
        end

        if isnothing(neighbor)
            # By default, only the order-1 and order+1 diagrams are considered to be the neighbors
            # Nd is the normalization diagram, by default, it only connects to the first diagram
            neighbor = Vector{Vector{Int}}([[d - 1, d + 1] for d = 1:Nd])
            neighbor[1] = (Nd == 2 ? [2,] : [Nd, 2]) # if Nd=2, then 2 must be the normalization diagram
            neighbor[end] = [1,] # norm to the first diag
            (Nd >= 3) && (neighbor[end-1] = [Nd - 2,]) # last diag to the second last, possible only for Nd>=3
        # elseif neighbor isa SimpleGraph
        #     @assert nv(neighbor) == Nd "The number of vertices in the neighbor graph should be equal to the number of integrands."
        #     neighbor = [neighbors(neighbor, ver) for ver in vertices(neighbor)]
        #     println(neighbor)
        elseif neighbor isa Vector{Tuple{Int,Int}}
            # @assert length(neighbor) == Nd "The number of neighbors should be equal to the number of integrands."
            g = SimpleGraph()
            add_vertices!(g, Nd)
            for n in neighbor
                add_edge!(g, n[1], n[2])
            end
            @assert is_connected(g) "The neighbor graph is not connected."
            neighbor = [neighbors(g, ver) for ver in vertices(g)]
        end
        @assert typeof(neighbor) == Vector{Vector{Int}} "Configuration.neighbor should be with a type of Vector{Vector{Int}} to avoid mistakes. Now get $(typeof(neighbor))"
        @assert Nd == length(neighbor) "$Nd elements are expected for neighbor=$neighbor"

        ############# initialize reweight factors ########################
        if isnothing(reweight)
            reweight = [1.0 for d = 1:Nd] # the last element is for the normalization diagram
        else
            push!(reweight, 1.0)
        end
        @assert Nd == length(reweight) "reweight vector size is wrong! Note that the last element in reweight vector is for the normalization diagram."

        if isnothing(seed)
            seed = rand(Random.RandomDevice(), 1:1000000)
        end
        rng = MersenneTwister(seed)

        curr = 1 # set the current diagram to be the first one
        norm = Nd
        # a small initial absweight makes the initial configuaration quickly updated,
        # so that no error is caused even if the intial absweight is wrong, 
        absweight = 1.0e-10
        normalization = 1.0e-10

        # visited[end] is for the normalization diagram
        visited = zeros(Float64, Nd) .+ 1.0e-8  # add a small initial value to avoid Inf when inverted

        # propose and accept shape: number of updates X integrand number X max(integrand number, variable number)
        # the last index will waste some memory, but the dimension is small anyway
        propose = zeros(Float64, (2, Nd, max(Nd, Nv))) .+ 1.0e-8 # add a small initial value to avoid Inf when inverted
        accept = zeros(Float64, (2, Nd, max(Nd, Nv)))

        return new{V,P,O}(seed, rng, para, totalStep, var,  # static parameters
            collect(neighbor), collect(dof), obs, collect(reweight), visited, # integrand properties
            0, curr, norm, normalization, absweight, propose, accept  # current MC state
        )
    end
end

function reset!(config, reweight=nothing)
    if typeof(config.observable) <: AbstractArray
        fill!(config.observable, zero(eltype(config.observable))) # reinialize observable
    else
        config.observable = zero(config.observable)
    end
    if isnothing(reweight) == false
        fill!(reweight, 1.0)
    end
    config.curr = 1
    config.normalization = 1.0e-10
    fill!(config.visited, 1.0e-8)
    fill!(config.propose, 1.0e-8)
    fill!(config.accept, 1.0e-10)
end

mutable struct FermiK{D} <: Variable
    # data::Vector{MVector{D,Float64}}
    data::Matrix{Float64}
    # data::Vector{Vector{Float64}}
    kF::Float64
    δk::Float64
    maxK::Float64
    offset::Int
    function FermiK(dim, kF, δk, maxK, size=MaxOrder; offset=0)
        @assert offset + 1 < size
        k = zeros(dim, size) .+ kF / sqrt(dim)
        # k0 = MVector{dim,Float64}([kF for i = 1:dim])
        # k0 = @SVector [kF for i = 1:dim]
        # k = [k0 for i = 1:size]
        return new{dim}(k, kF, δk, maxK, offset)
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
    function RadialFermiK(kF=1.0, δk=0.01, size=MaxOrder; offset=0)
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
    function Tau(β=1.0, λ=0.5, size=MaxOrder; offset=0)
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
    function Continuous(bound, λ=nothing, size=MaxOrder; offset=0)
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
    function Angle(λ=0.5, size=MaxOrder; offset=0)
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
    function TauPair(β=1.0, λ=0.5, size=MaxOrder; offset=0)
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
    function Discrete(lower, upper, size=MaxOrder; offset=0)
        d = [i for i = 1:size] #avoid dulication
        @assert offset + 1 < size
        @assert upper > lower
        return new(d, lower, upper, upper - lower + 1, offset)
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
