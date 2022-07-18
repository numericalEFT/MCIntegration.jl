"""
    mutable struct Configuration

    Struct that contains everything needed for MC.

There are three different componenets:

 # Members

 ## Static parameters

 - `seed`: seed to initialize random numebr generator, also serves as the unique pid of the configuration

 - `rng`: a MersenneTwister random number generator, seeded by `seed`

 - `para`: user-defined parameter, set to nothing if not needed

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

-  `reweight_goal`: The expected distribution of visited times for each integrand after reweighting . If not set, then all factors will be initialized with one.

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
    var::V

    ########### integrand properties ##############
    neighbor::Vector{Vector{Int}}
    dof::Vector{Vector{Int}} # degrees of freedom
    observable::O  # observables for each integrand
    reweight::Vector{Float64}
    reweight_goal::Vector{Float64}
    visited::Vector{Float64}

    ############# current state ######################
    neval::Int64 # number of evaluations performed up to now
    curr::Int # index of current integrand
    norm::Int # index of the normalization diagram
    normalization::Float64 # normalization factor for observables
    absWeight::Float64 # the absweight of the current diagrams. Store it for fast updates

    propose::Array{Float64,3} # updates index, integrand index, integrand index
    accept::Array{Float64,3} # updates index, integrand index, integrand index 

    """
    function Configuration(var::V, dof, obs::O=length(dof) == 1 ? 0.0 : zeros(length(dof));
        para::P=nothing,
        reweight::Vector{Float64}=ones(length(dof) + 1),
        seed::Int=rand(Random.RandomDevice(), 1:1000000),
        neighbor::Union{Vector{Vector{Int}},Vector{Tuple{Int,Int}},Nothing}=nothing
    ) where {V,P,O}

Create a Configuration struct

    # Arguments

    ## Static parameters

    - `var`: TUPLE of variables, each variable should be derived from the abstract type Variable, see variable.jl for details). Use a tuple rather than a vector improves the performance.

    - `dof::Vector{Vector{Int}}`: degrees of freedom of each integrand, e.g., [[0, 1], [2, 3]] means the first integrand has zero var#1 and one var#2; while the second integrand has two var#1 and 3 var#2. 

    - `obs`: observables that is required to calculate the integrands, will be used in the `measure` function call.
    It is either an array of any type with the common operations like +-*/^ defined. 
    By default, it will be set to 0.0 if there is only one integrand (e.g., length(dof)==1); otherwise, it will be set to zeros(length(dof)).

    - `para`: user-defined parameter, set to nothing if not needed

    - `reweight`: reweight factors for each integrands. If not set, then all factors will be initialized with one.

    - `reweight_goal`: The expected distribution of visited times for each integrand after reweighting . If not set, then all factors will be initialized with one.

    - `seed`: seed to initialize random numebr generator, also serves as the unique pid of the configuration. If it is nothing, then use RandomDevice() to generate a random seed in [1, 1000_1000]

- `neighbor::Vector{Tuple{Int, Int}}` : vector of tuples that defines the neighboring integrands. Two neighboring integrands are directly connected in the Markov chain. 
    e.g., [(1, 2), (2, 3)] means the integrand 1 and 2 are neighbor, and 2 and 3 are neighbor.  
    The neighbor vector defines a undirected graph showing how the integrands are connected. Please make sure all integrands are connected.
    By default, we assume the N integrands are in the increase order, meaning the neighbor will be set to [(N+1, 1), (1, 2), (2, 4), ..., (N-1, N)], where the first N entries are for diagram 1, 2, ..., N and the last entry is for the normalization diagram. Only the first diagram is connected to the normalization diagram.
    Only highly correlated integrands are not highly correlated should be defined as neighbors. Otherwise, most of the updates between the neighboring integrands will be rejected and wasted.
    """
    function Configuration(var::V, dof, obs::O=length(dof) == 1 ? 0.0 : zeros(length(dof));
        para::P=nothing,
        reweight::Vector{Float64}=ones(length(dof) + 1),
        reweight_goal::Vector{Float64}=ones(length(dof) + 1),
        seed::Int=rand(Random.RandomDevice(), 1:1000000),
        neighbor::Union{Vector{Vector{Int}},Vector{Tuple{Int,Int}},Nothing}=nothing
    ) where {V,P,O}
        # println(var)
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
        @assert Nd == length(reweight) "reweight vector size is wrong! Note that the last element in reweight vector is for the normalization diagram."
        @assert Nd == length(reweight_goal) "reweight_goal vector size is wrong! Note that the last element in reweight vector is for the normalization diagram."
        @assert all(x -> x > 0, reweight) "All reweight factors should be positive."
        @assert all(x -> x > 0, reweight_goal) "All reweight_goal factors should be positive."
        reweight .*= reweight_goal
        reweight /= sum(reweight) # normalize the reweight factors

        curr = 1 # set the current diagram to be the first one
        norm = Nd # set the normalization diagram to be the last one
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

        return new{V,P,O}(seed, MersenneTwister(seed), para, var,  # static parameters
            collect(neighbor), collect(dof), obs, collect(reweight), collect(reweight_goal),
            visited, # integrand properties
            0, curr, norm, normalization, absweight, propose, accept  # current MC state
        )
    end
end

function clearStatistics!(config)
    if typeof(config.observable) <: AbstractArray
        fill!(config.observable, zero(eltype(config.observable))) # reinialize observable
    else
        config.observable = zero(config.observable)
    end
    # if isnothing(reweight) == false
    #     fill!(reweight, 1.0)
    # end
    config.neval = 0
    config.curr = 1
    config.normalization = 1.0e-10
    fill!(config.visited, 1.0e-8)
    fill!(config.propose, 1.0e-8)
    fill!(config.accept, 1.0e-10)
    for var in config.var
        clearStatistics!(var)
    end
end

function addConfig!(c::Configuration, ic::Configuration)
    c.visited += ic.visited
    c.accept += ic.accept
    c.propose += ic.propose
    c.neval += ic.neval
    c.normalization += ic.normalization
    c.observable += ic.observable
    for (vi, var) in enumerate(c.var)
        addStatistics!(var, ic.var[vi])
    end
end

function MPIreduceConfig!(c::Configuration, root, comm)

    ########## variable that could be a number ##############
    neval = MPI.Reduce(c.neval, MPI.SUM, root, comm)
    normalization = MPI.Reduce(c.normalization, MPI.SUM, root, comm)
    observable = MPI.Reduce(c.observable, MPI.SUM, root, comm)
    if MPI.Comm_rank(comm) == root
        c.neval = neval
        c.normalization = normalization
        c.observable = observable
    end
    for vi in 1:length(c.var)
        histogram = MPI.Reduce(c.var[vi].histogram, MPI.SUM, root, comm)
        if MPI.Comm_rank(comm) == root
            c.var[vi].histogram = histogram
        end
    end

    ########## variable that are vectors ##############
    MPI.Reduce!(c.visited, MPI.SUM, root, comm)
    MPI.Reduce!(c.propose, MPI.SUM, root, comm)
    MPI.Reduce!(c.accept, MPI.SUM, root, comm)
    # if MPI.Comm_rank(comm) == root
    #     # reweight ./= MPI.Comm_size(comm)
    #     # return SummaryStat(neval, visited, reweight, propose, accept)
    #     rc = deepcopy(c)
    #     rc.neval = MPI.Reduce(c.neval, MPI.SUM, root, comm)
    #     rc.visited = MPI.Reduce(c.visited, MPI.SUM, root, comm)
    #     rc.propose = MPI.Reduce(c.propose, MPI.SUM, root, comm)
    #     rc.accept = MPI.Reduce(c.accept, MPI.SUM, root, comm)
    #     rc.observable = MPI.Reduce(c.observable, MPI.SUM, root, comm)
    #     rc.normalization = MPI.Reduce(c.normalization, MPI.SUM, root, comm)
    #     return rc
    # else
    #     MPI.Reduce(c.neval, MPI.SUM, root, comm)
    #     MPI.Reduce(c.visited, MPI.SUM, root, comm)
    #     MPI.Reduce(c.propose, MPI.SUM, root, comm)
    #     MPI.Reduce(c.accept, MPI.SUM, root, comm)
    #     MPI.Reduce(c.observable, MPI.SUM, root, comm)
    #     MPI.Reduce(c.normalization, MPI.SUM, root, comm)
    #     return c
    # end
end

function summary(config::Configuration, total_neval=nothing)
    neval, visited, reweight, propose, accept = config.neval, config.visited, config.reweight, config.propose, config.accept
    var, neighbor = config.var, config.neighbor

    Nd = length(visited)

    barbar = "===========================  Configuration  ========================================="
    bar = "-------------------------------------------------------------------------------------"

    println(barbar)
    println(green(Dates.now()))
    println("\nneval = $(config.neval)")
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
            typestr = "$(typeof(var))"
            typestr = split(typestr, ".")[end]
            @printf(
                "  %2d / %-10s:   %11.6f%% %11.6f%% %12.6f\n",
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
    println(yellow("Total Proposed: $(totalproposed / neval * 100.0)%\n"))
    if isnothing(total_neval) == false
        println(green(progressBar(neval, total_neval)))
    end
    println()

end