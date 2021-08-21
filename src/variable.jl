abstract type Variable end
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
 
- `neighbor::Vector{Vector{Int}}` : vectors that indicates the neighbors of each integrand. e.g., ([2, ], [1, ]) means the neighbor of the first integrand is the second one, while the neighbor of the second integrand is the first. 
    There is a MC update proposes to jump from one integrand to another. If these two integrands' degrees of freedom are very different, then the update is unlikely to be accepted. To avoid this problem, one can specify neighbor to guide the update. 
    
    By default, we assume the N integrands are in the increase order, meaning the neighbor will be set to ([N+1, 2], [1, 3], [2, 4], ..., [N-1,], [1, ]), where the first N entries are for diagram 1, 2, ..., N and the last entry is for the normalization diagram. Only the first diagram is connected to the normalization diagram.

 - `dof::Vector{Vector{Int}}`: degrees of freedom of each integrand, e.g., [[0, 1], [2, 3]] means the first integrand has zero var#1 and one var#2; while the second integrand has two var#1 and 3 var#2. 

 - `observable`: observables that is required to calculate the integrands, will be used in the `measure` function call.
    It is either an array of any type with the common operations like +-*/^ defined. 

 - `reweight`: reweight factors for each integrands. 

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

 - `totalStep`: the total number of updates for this configuration

 - `var`: TUPLE of variables, each variable should be derived from the abstract type Variable, see variable.jl for details). Use a tuple rather than a vector improves the performance.

 - `dof::Vector{Vector{Int}}`: degrees of freedom of each integrand, e.g., [[0, 1], [2, 3]] means the first integrand has zero var#1 and one var#2; while the second integrand has two var#1 and 3 var#2. 

 - `obs`: observables that is required to calculate the integrands, will be used in the `measure` function call
    It is either an array of any type with the common operations like +-*/^ defined. 

 - `para`: user-defined parameter, set to nothing if not needed

 - `reweight`: reweight factors for each integrands. If not set, then all factors will be initialized with one.

 - `seed`: seed to initialize random numebr generator, also serves as the unique pid of the configuration. If it is nothing, then use RandomDevice() to generate a random seed in [1, 1000_1000]

- `neighbor::Vector{Vector{Int}}` : vectors that indicates the neighbors of each integrand. e.g., ([2, ], [1, ]) means the neighbor of the first integrand is the second one, while the neighbor of the second integrand is the first. 
    There is a MC update proposes to jump from one integrand to another. If these two integrands' degrees of freedom are very different, then the update is unlikely to be accepted. To avoid this problem, one can specify neighbor to guide the update. 
    
    By default, we assume the N integrands are in the increase order, meaning the neighbor will be set to ([N+1, 2], [1, 3], [2, 4], ..., [N-1,], [1, ]), where the first N entries are for diagram 1, 2, ..., N and the last entry is for the normalization diagram. Only the first diagram is connected to the normalization diagram.
"""
    function Configuration(totalStep, var::V, dof, obs::O; para::P=nothing, reweight=nothing, seed=nothing, neighbor=Vector{Vector{Int}}([])) where {V,P,O}
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

        if neighbor == []
            # By default, only the order-1 and order+1 diagrams are considered to be the neighbors
            # Nd is the normalization diagram, by default, it only connects to the first diagram
            neighbor = Vector{Vector{Int}}([[d - 1, d + 1] for d in 1:Nd])
            neighbor[1] = (Nd == 2 ? [2, ] : [Nd, 2]) # if Nd=2, then 2 must be the normalization diagram
            neighbor[end] = [1, ] # norm to the first diag
            (Nd >= 3) && (neighbor[end - 1] = [Nd - 2, ]) # last diag to the second last, possible only for Nd>=3
        end
        @assert typeof(neighbor) == Vector{Vector{Int}} "Configuration.neighbor should be with a type of Vector{Vector{Int}} to avoid mistakes. Now get $(typeof(neighbor))"
        @assert Nd == length(neighbor) "$Nd elements are expected for neighbor=$neighbor"

        ############# initialize reweight factors ########################
        if isnothing(reweight)
            reweight = [1.0 for d in 1:Nd] # the last element is for the normalization diagram
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
    data::Vector{SVector{D,Float64}}
    # data::Vector{Vector{Float64}}
    kF::Float64
    δk::Float64
    maxK::Float64
    function FermiK(dim, kF, δk, maxK, size=MaxOrder)
        k0 = SVector{dim,Float64}([kF for i = 1:dim])
        # k0 = @SVector [kF for i = 1:dim]
        k = [k0 for i in 1:size]
        return new{dim}(k, kF, δk, maxK)
    end
end

mutable struct RadialFermiK <: Variable
    data::Vector{Float64}
    kF::Float64
    δk::Float64
    function RadialFermiK(kF=1.0, δk=0.01, size=MaxOrder)
        k = [kF for i = 1:size]
        return new(k, kF, δk)
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
    function Tau(β=1.0, λ=0.5, size=MaxOrder)
        t = [β / 2.0 for i = 1:size]
        return new(t, λ, β)
    end
end

mutable struct Angle <: Variable
    data::Vector{Float64}
    λ::Float64
    function Angle(λ=0.5, size=MaxOrder)
        theta = [π for i = 1:size]
        return new(theta, λ)
    end
end


mutable struct TauPair <: Variable
    data::Vector{MVector{2,Float64}}
    λ::Float64
    β::Float64
    function TauPair(β=1.0, λ=0.5, size=MaxOrder)
        t = [@MVector [β / 3.0, β / 2.0] for i = 1:size]
        return new(t, λ, β)
    end
end

mutable struct Discrete <: Variable
    data::Vector{Int}
    lower::Int
    upper::Int
    size::Int
    function Discrete(lower, upper, size=MaxOrder)
        d = [1 for i in 1:size]
        @assert upper > lower
        return new(d, lower, upper, upper - lower + 1)
    end
end


Base.getindex(Var::Variable, i::Int) = Var.data[i]
function Base.setindex!(Var::Variable, v, i::Int)
    Var.data[i] = v
end
Base.firstindex(Var::Variable) = 1 # return index, not the value
Base.lastindex(Var::Variable) = length(Var.data) # return index, not the value
