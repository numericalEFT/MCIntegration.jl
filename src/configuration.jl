"""
    mutable struct Configuration{NI,V,P,O,T}

    Struct that contains everything needed for MC.

# Parameters
- `NI` : number of integrands
- `V` : type of variables
- `P` : type of user-defined data
- `O` : type of observables 
- `T` : type of integrand

# Static parameters
- `seed`: seed to initialize random numebr generator, also serves as the unique pid of the configuration
- `rng`: a MersenneTwister random number generator, seeded by `seed`
- `userdata`: user-defined parameter
- `var`: TUPLE of variables, each variable should be derived from the abstract type Variable, see variable.jl for details). Use a tuple rather than a vector improves the performance.

# integrand properties
- `neighbor::Vector{Tuple{Int, Int}}` : vector of tuples that defines the neighboring integrands. Two neighboring integrands are directly connected in the Markov chain. 
    e.g., [(1, 2), (2, 3)] means the integrand 1 and 2 are neighbor, and 2 and 3 are neighbor.  
   The neighbor vector defines a undirected graph showing how the integrands are connected. Please make sure all integrands are connected.
   By default, we assume the N integrands are in the increase order, meaning the neighbor will be set to [(N+1, 1), (1, 2), (2, 3), ..., (N-1, N)], where the first N entries are for diagram 1, 2, ..., N and the last entry is for the normalization diagram. Only the first diagram is connected to the normalization diagram.
   Only highly correlated integrands are not highly correlated should be defined as neighbors. Otherwise, most of the updates between the neighboring integrands will be rejected and wasted.
- `dof::Vector{Vector{Int}}`: degrees of freedom of each integrand, e.g., [[0, 1], [2, 3]] means the first integrand has zero var#1 and one var#2; while the second integrand has two var#1 and 3 var#2. 
- `observable`: observables that is required to calculate the integrands, will be used in the `measure` function call.
    It is either an array of any type with the common operations like +-*/^ defined. 
- `reweight`: reweight factors for each integrands. The reweight factor of the normalization diagram is assumed to be 1. Note that you don't need to explicitly add the normalization diagram. 
- `visited`: how many times this integrand is visited by the Markov chain.

# current MC state
- `step`: the number of MC updates performed up to now
- `norm`: the index of the normalization diagram. `norm` is larger than the index of any user-defined integrands 
- `normalization`: the accumulated normalization factor. Physical observable = Configuration.observable/Configuration.normalization.
- `propose/accept`: array to store the proposed and accepted updates for each integrands and variables.
    Their shapes are (number of updates X integrand number X max(integrand number, variable number).
    The last index will waste some memory, but the dimension is small anyway.
"""

"""
    mutable struct Configuration{NI,V,P,O,T}

Struct that holds all the necessary parameters and variables for Monte Carlo integration.

# Parameters

- `NI` : Number of integrands
- `V` : Type of variables
- `P` : Type of user-defined data
- `O` : Type of observables 
- `T` : Type of integrand

# Static parameters

- `seed`: Seed to initialize the random number generator, also serves as the unique process ID of the configuration.
- `rng`: A MersenneTwister random number generator, seeded by `seed`.
- `userdata`: User-defined parameter.
- `var`: Tuple of variables. Each variable should derive from the abstract type `Variable` (see `variable.jl` for details). Using a tuple instead of a vector improves performance.

# Integrands properties

- `neighbor::Vector{Tuple{Int, Int}}` : Vector of tuples defining neighboring integrands. Two neighboring integrands are directly connected in the Markov chain. 
   The `neighbor` vector defines an undirected graph showing how the integrands are connected. Only highly correlated integrands should be defined as neighbors to reduce autocorrelations.
- `dof::Vector{Vector{Int}}`: Degrees of freedom of each integrand, i.e., the dimensions in which each integrand can vary.
- `observable`: Observables required to calculate the integrands, will be used in the `measure` function call.
- `reweight`: Reweight factors for each integrand. The reweight factor of the normalization integrand (namely, the last element) is assumed to be 1.
- `visited`: The number of times each integrand is visited by the Markov chain.

# Current MC state

- `step`: The number of Monte Carlo updates performed up to now.
- `norm`: The index of the normalization integrand. `norm` is larger than the index of any user-defined integrands.
- `normalization`: The accumulated normalization factor.
- `propose/accept`: Arrays to store the proposed and accepted updates for each integrand and variable.
"""
mutable struct Configuration{NI,V,P,O,T}
    ########### static parameters ###################
    seed::Int # seed to initialize random numebr generator, also serves as the unique pid of the configuration
    rng::MersenneTwister # random number generator seeded by seed
    var::V
    userdata::P

    ########### integrand properties ##############
    N::Int # number of integrands (does not include the normalization diagram)
    neighbor::Vector{Vector{Int}}
    dof::Vector{Vector{Int}} # degrees of freedom
    maxdof::Vector{Int} # max degrees of freedom of all integrands, length(maxdof) = length(var)
    observable::O  # observables for each integrand
    reweight::Vector{Float64}
    visited::Vector{Float64}

    ############# current state ######################
    neval::Int # MC steps
    norm::Int # index of the normalization diagram
    normalization::Float64 # normalization factor for observables

    propose::Array{Float64,3} # updates index, integrand index, integrand index
    accept::Array{Float64,3} # updates index, integrand index, integrand index 
end

function Base.show(io::IO, config::Configuration)
    print(io,
        "Configuration for $(config.N) integrands involves $(length(config.var)) types of variables.\nNumber of variables for each integrand: $(config.dof[1:end-1]).\n"
    )
end

"""
    function Configuration(;
        var::Union{Variable,AbstractVector,Tuple}=(Continuous(0.0, 1.0),),
        dof::Union{Int,AbstractVector,AbstractMatrix}=[ones(Int, length(var)),],
        type=Float64,  # type of the integrand
        obs::AbstractVector=zeros(type, length(dof)),
        reweight::Vector{Float64}=ones(length(dof) + 1),
        seed::Int=rand(Random.RandomDevice(), 1:1000000),
        neighbor::Union{Vector{Vector{Int}},Vector{Tuple{Int,Int}},Nothing}=nothing,
        userdata=nothing,
        kwargs...
    )

Create a Configuration struct

# Arguments
- `var`: TUPLE of variables, each variable should be derived from the abstract type Variable, see variable.jl for details). Use a tuple rather than a vector improves the performance.
By default, var = (Continuous(0.0, 1.0),), which is a single continuous variable.
- `dof::Vector{Vector{Int}}`: degrees of freedom of each integrand, e.g., [[0, 1], [2, 3]] means the first integrand has zero var#1 and one var#2; while the second integrand has two var#1 and 3 var#2. 
By default, dof=[ones(length(var)), ], which means that there is only one integrand, and each variable has one degree of freedom.
- `obs`: observables that is required to calculate the integrands, will be used in the `measure` function call.
It is either an array of any type with the common operations like +-*/^ defined. 
By default, it will be set to 0.0 if there is only one integrand (e.g., length(dof)==1); otherwise, it will be set to zeros(length(dof)).
- `para`: user-defined parameter, set to nothing if not needed
- `reweight`: reweight factors for each integrands. If not set, then all factors will be initialized with one. Internally, a reweight factor of 1 will be appended to the end of the reweight vector, which is for the normalization integral.
- `seed`: seed to initialize random numebr generator, also serves as the unique `pid` of the configuration. If it is `nothing`, then use `RandomDevice()` to generate a random seed in `[1, 1000_1000]`
- `neighbor::Vector{Tuple{Int, Int}}` : vector of tuples that defines the neighboring integrands. Two neighboring integrands are directly connected in the Markov chain. 
    e.g., [(1, 2), (2, 3)] means the integrand 1 and 2 are neighbor, and 2 and 3 are neighbor.  
    The neighbor vector defines a undirected graph showing how the integrands are connected. Please make sure all integrands are connected.
    By default, we assume the N integrands are in the increase order, meaning the neighbor will be set to [(N+1, 1), (1, 2), (2, 4), ..., (N-1, N)], where the first N entries are for integrals 1, 2, ..., N and the last entry is for the normalization integral. Only the first integral is connected to the normalization integral.
    Only highly correlated integrands should be defined as neighbors. Otherwise, most of the updates between the neighboring integrands will be rejected and wasted.
- `userdata`: User data you want to pass to the integrand and the measurement
"""

"""
    function Configuration(;
        var::Union{Variable,AbstractVector,Tuple}=(Continuous(0.0, 1.0),),
        dof::Union{Int,AbstractVector,AbstractMatrix}=[ones(Int, length(var))],
        type=Float64,  # type of the integrand
        obs::AbstractVector=zeros(Float64, length(dof)),
        reweight::Vector{Float64}=ones(length(dof) + 1),
        seed::Int=rand(Random.RandomDevice(), 1:1000000),
        userdata=nothing,
        neighbor::Union{Vector{Vector{Int}},Vector{Tuple{Int,Int}},Nothing}=nothing,
        kwargs...
    )

Create a Configuration struct for MC integration.

# Arguments
- `var`: Either a single `Variable`, a `CompositeVar`, or a tuple consisting of `Variable` and/or `CompositeVar` instances. Tuples are used to improve performance by ensuring type stability. By default, `var` is set to a tuple containing a single continuous variable, `(Continuous(0.0, 1.0),)`.
- `dof`: Degrees of freedom for each integrand, as a vector of integers. For example, `[[0, 1], [2, 3]]` means the first integrand has zero instances of var#1 and one of var#2; while the second integrand has two instances of var#1 and 3 of var#2. Defaults to `[ones(length(var))]`, i.e., one degree of freedom for each variable.
- `type`: Type of the integrand, Float64 by default.
- `obs`: Vector of observables needed to calculate the integrands, which will be used in the `measure` function call. 
- `reweight`: Vector of reweight factors for each integrand. By default, all factors are initialized to one. Internally, a reweight factor of 1 will be appended to the end of the reweight vector, which is for the normalization integral.
- `seed`: Seed for the random number generator. This also serves as the unique identifier of the configuration. If it is `nothing`, then a random seed between 1 and 1,000,000 is generated. 
- `userdata`: User data to pass to the integrand and the measurement.
- `neighbor`: Vector of tuples that define neighboring integrands. For example, `[(1, 2), (2, 3)]` means that the first and second integrands, and the second and third integrands, are neighbors. Neighboring integrands are directly connected in the Markov chain. By default, all integrands are connected in ascending order. Note that the normalization integral is automatically appended at the end of the integrand list and is considered as neighbor with the first user-defined integrand.

# Example
```julia
cfg = Configuration(
    var = (Continuous(0.0, 1.0), Continuous(-1.0, 1.0)),
    dof = [[1, 1], [2, 0]],
    obs = [0.0, 0.0],
    seed = 1234,
    neighbor = [(1, 2)]
)
```
"""
function Configuration(;
    var::V=(Continuous(0.0, 1.0),),
    dof::Union{Int,AbstractVector,AbstractMatrix}=(is_variable(typeof(var)) ? 1 : [ones(Int, length(var)),]),
    type=Float64,  # type of the integrand
    obs::AbstractVector=zeros(type, length(dof)),
    reweight::Vector{Float64}=ones(length(dof) + 1),
    seed::Int=rand(Random.RandomDevice(), 1:1000000),
    neighbor::Union{Vector{Vector{Int}},Vector{Tuple{Int,Int}},Nothing}=nothing,
    userdata=nothing,
    kwargs...
) where {V}
    if is_variable(V)
        var = (var,)
    elseif (V <: AbstractVector) || (V <: Tuple)
        @assert all(v -> is_variable(typeof(v)), var) "All elements in var should be derived from the abstract type Variable, now got $(is_variable.(var))"
        if var isa AbstractVector
            var = Tuple(v for v in var)
        end
    else
        error("Configuration.var should be a variable, a vector of variables, or a tuple of variables. Now get $(V)")
    end
    @assert (var isa Tuple) "Failed to convet Configuration.var to a tuple of Variable to maximize efficiency. Now get $(typeof(V))"
    if isconcretetype(typeof(var)) == false
        @warn "Type of Configuration.var is $(typeof(var)), which is not a concrete type. This may cause performance issue."
    end

    Nv = length(var) # number of variables

    ################# integrand initialization #########################
    if dof isa Int  #one integral with one variable
        @assert length(var) == 1 "Only one type of variable is allowed when dof is an integer"
        dof = [[dof,],]
        # elseif (dof isa AbstractVector) && eltype(dof) <: Int #one integral with multiple variables
        #     dof = [deepcopy(dof),]
    elseif dof isa AbstractMatrix #each column is a dof for one integral
        dof = [dof[:, i] for i in 1:size(dof)[2]]
    else
        if eltype(dof) <: Tuple{Vararg{Any}}
            dof = [collect(d) for d in dof]
        elseif eltype(dof) <: AbstractVector
            dof = deepcopy(dof) # don't modify the input dof
        elseif eltype(dof) <: Int
            dof = [[d,] for d in dof]
        else
            error("Configuration.dof should be a Vector{Int} or Tuple{Int, ..., Int} or Vector{Vector{Int}} or Vector{Tuple{Int, ..., Int}} to avoid mistakes. Now get $(typeof(dof))")
        end
    end
    # add normalization diagram to dof
    push!(dof, zeros(Int, length(var))) # add the degrees of freedom for the normalization diagram

    maxdof = _maxdof(dof)
    for i in eachindex(maxdof)
        if maxdof[i] + var[i].offset >= poolsize(var[i]) - 2 # two more elements for caching purpose
            resize!(var[i], maxdof[i] + 2 + var[i].offset)
        end
    end

    Nd = length(dof) # number of integrands + renormalization diagram
    @assert Nd > 1 "At least one integrand is required."
    # make sure dof has the correct size that matches var and neighbor
    # println(Nv, ", ", dof)
    @assert all(nv -> length(nv) == Nv, dof) "Each element of `dof` should have the same dimension as `var`"

    @assert length(obs) == Nd - 1 "The number of observables should be equal to the number of integrands"

    neighbor = _neighbor(neighbor, Nd)

    reweight = collect(reweight)
    reweight /= sum(reweight) # normalize the reweight factors
    @assert Nd == length(reweight) "Wrong reweight vector size! Note that the last element in reweight vector is for the normalization diagram."
    @assert all(x -> x > 0, reweight) "All reweight factors should be positive."

    norm = Nd # set the normalization diagram to be the last one
    # so that no error is caused even if the intial absweight is wrong, 
    normalization = 1.0e-10

    # visited[end] is for the normalization diagram
    visited = zeros(Float64, Nd) .+ 1.0e-8  # add a small initial value to avoid Inf when inverted

    # propose and accept shape: number of updates X integrand number X max(integrand number, variable number)
    # the last index will waste some memory, but the dimension is small anyway
    propose = zeros(Float64, (2, Nd, max(Nd, Nv))) .+ 1.0e-8 # add a small initial value to avoid Inf when inverted
    accept = zeros(Float64, (2, Nd, max(Nd, Nv)))

    return Configuration{Nd - 1,typeof(var),typeof(userdata),typeof(obs),type}(
        seed, MersenneTwister(seed), var, userdata,  # static parameters
        Nd - 1, neighbor, dof, maxdof, obs, reweight, visited, # integrand properties
        0, norm, normalization, propose, accept  # current MC state
    )
end

function reset_seed!(cfg::Configuration, seed::Int)
    cfg.seed = seed
    cfg.rng = MersenneTwister(seed)
end

function _neighbor(neighbor, Nd)
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
    neighbor = collect(neighbor)
    @assert neighbor isa Vector{Vector{Int}} "Configuration.neighbor should be with a type of Vector{Vector{Int}} to avoid mistakes. Now get $(typeof(neighbor))"
    @assert Nd == length(neighbor) "$Nd elements are expected for neighbor=$neighbor"
    return neighbor
end

function _maxdof(dof)
    dof = collect(dof)
    maxdof = zeros(Int, length(dof[1]))
    for i in eachindex(maxdof)
        maxdof[i] = maximum([dof[j][i] for j in eachindex(dof)])
    end
    return maxdof
end

function clearStatistics!(config)
    for i in eachindex(config.observable)
        config.observable[i] = zero(config.observable[i])
    end
    config.neval = 0
    config.normalization = 1.0e-10
    fill!(config.visited, 1.0e-8)
    fill!(config.propose, 1.0e-8)
    fill!(config.accept, 1.0e-10)
    for var in config.var
        Dist.clearStatistics!(var)
    end
end

function addConfig!(c::Configuration, ic::Configuration)
    c.visited += ic.visited
    c.accept += ic.accept
    c.propose += ic.propose
    c.neval += ic.neval
    c.normalization += ic.normalization
    c.observable .+= ic.observable
    for (vi, var) in enumerate(c.var)
        Dist.addStatistics!(var, ic.var[vi])
    end
end

function MPIreduceConfig!(c::Configuration, root=0, comm=MPI.COMM_WORLD)
    # Need to reduce from workers:
    # neval
    # var.histogram
    # visited, propose, accept
    # normalization, observable

    function histogram_reduce!(var)
        if var isa Dist.CompositeVar
            for v in var.vars
                histogram_reduce!(v)
            end
        else
            MCUtility.MPIreduce!(var.histogram)
        end
    end

    ########## variable that could be a number ##############
    c.neval = MCUtility.MPIreduce(c.neval) # reduce the amount of the commuication
    c.normalization = MCUtility.MPIreduce(c.normalization) # reduce the amount of the commuication
    for o in eachindex(c.observable)
        if c.observable[o] isa AbstractArray
            MCUtility.MPIreduce!(c.observable[o]) # avoid memory allocation
        else
            c.observable[o] = MCUtility.MPIreduce(c.observable[o])
        end
    end
    for v in c.var
        histogram_reduce!(v)
    end

    ########## variable that are vectors ##############
    MCUtility.MPIreduce!(c.visited)
    MCUtility.MPIreduce!(c.propose)
    MCUtility.MPIreduce!(c.accept)
end

function MPIbcastConfig!(c::Configuration, root=0, comm=MPI.COMM_WORLD)
    # need to broadcast from root to workers:
    # reweight
    # var.histogram
    function histogram_bcast!(var)
        if var isa Dist.CompositeVar
            for v in var.vars
                histogram_bcast!(v)
            end
        else
            MCUtility.MPIbcast!(var.histogram)
        end
    end

    ########## variable that could be a number ##############
    MCUtility.MPIbcast(c.reweight)

    for v in c.var
        histogram_bcast!(v)
    end
end

function bcastConfig!(dest::Configuration, src::Configuration)
    # need to broadcast from root to workers:
    # reweight
    # var.histogram
    ########## variable that could be a number ##############
    dest.reweight .= src.reweight

    function histogram_bcast!(dest, src)
        if dest isa Dist.CompositeVar
            for i in 1:length(dest.vars)
                histogram_bcast!(dest.vars[i], src.vars[i])
            end
        else
            dest.histogram .= src.histogram
        end
    end

    for i in 1:length(dest.var)
        histogram_bcast!(dest.var[i], src.var[i])
    end
end

function report(config::Configuration, total_neval=nothing)
    neval, visited, reweight, propose, accept = config.neval, config.visited, config.reweight, config.propose, config.accept
    var, neighbor = config.var, config.neighbor

    Nd = length(visited)

    barbar = "===========================  Configuration  ========================================="
    bar = "-------------------------------------------------------------------------------------"

    println()
    println(barbar)
    println(green(Dates.now()))
    println(bar)
    println(yellow("Integral num = $(config.N), dof = $(config.dof), with variables:"))
    # println("\nneval = $(config.neval)")
    for (vi, var) in enumerate(config.var)
        println("$vi. $var")
    end
    println(bar)

    totalproposed = 0.0
    println(yellow(@sprintf("%-20s %12s %12s %12s", "ChangeIntegrand", "Proposed", "Accepted", "Ratio  ")))
    for n in neighbor[Nd]
        @printf(
            "Norm -> %2d:           %11.6f%% %11.6f%% %12.6f\n",
            n,
            propose[1, Nd, n] / neval * 100.0,
            accept[1, Nd, n] / neval * 100.0,
            accept[1, Nd, n] / propose[1, Nd, n]
        )
        totalproposed += propose[1, Nd, n]
    end
    for idx = 1:Nd-1
        for n in neighbor[idx]
            if n == Nd  # normalization diagram
                @printf("  %d ->Norm:           %11.6f%% %11.6f%% %12.6f\n",
                    idx,
                    propose[1, idx, n] / neval * 100.0,
                    accept[1, idx, n] / neval * 100.0,
                    accept[1, idx, n] / propose[1, idx, n]
                )
            else
                @printf("  %d -> %2d:            %11.6f%% %11.6f%% %12.6f\n",
                    idx, n,
                    propose[1, idx, n] / neval * 100.0,
                    accept[1, idx, n] / neval * 100.0,
                    accept[1, idx, n] / propose[1, idx, n]
                )
            end
            totalproposed += propose[1, idx, n]
        end
    end
    println(bar)

    println(yellow(@sprintf("%-20s %12s %12s %12s", "ChangeVariable", "Proposed", "Accepted", "Ratio  ")))
    for idx = 1:Nd-1 # normalization diagram don't have variable to change
        for (vi, var) in enumerate(var)
            if var isa Continuous
                typestr = "Continuous"
            elseif var isa Discrete
                typestr = "Discrete"
            elseif var isa CompositeVar
                typestr = "Composite"
            elseif var isa FermiK
                typestr = "FermiK"
            else
                typestr = "$(typeof(var))"
                # typestr = split(typestr, ".")[end]
            end
            @printf(
                "  %2d / %-11s:   %11.6f%% %11.6f%% %12.6f\n",
                idx, typestr,
                propose[2, idx, vi] / neval * 100.0,
                accept[2, idx, vi] / neval * 100.0,
                accept[2, idx, vi] / propose[2, idx, vi]
            )
            totalproposed += propose[2, idx, vi]
        end
    end
    println(bar)
    println(yellow("Integrand            Visited      ReWeight"))
    @printf("  Norm   :     %12i %12.6f\n", visited[end], reweight[end])
    for idx = 1:Nd-1
        @printf("  Order%2d:     %12i %12.6f\n", idx, visited[idx], reweight[idx])
    end
    println(bar)
    println(yellow("Integrand evaluation = $neval\n"))
    # if isnothing(total_neval) == false
    #     println(yellow(progressBar(neval, total_neval)))
    # end
    # println()

end