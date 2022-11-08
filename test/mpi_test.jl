using Test
using MPI
using MCIntegration
const MCUtility = MCIntegration.MCUtility

@testset "MPI reduce" begin
    (MPI.Initialized() == false) && MPI.Init()
    comm = MPI.COMM_WORLD
    Nworker = MPI.Comm_size(comm)  # number of MPI workers
    rank = MPI.Comm_rank(comm)  # rank of current MPI worker
    root = 0 # rank of the root worker

    a = [1, 2, 3] 
    a = MCUtility.MPIreduce(a)
    if rank == root
        @test a == [Nworker, 2Nworker, 3Nworker]
    end

    b = 1
    b = MCUtility.MPIreduce(b)
    if rank == root
        @test b == Nworker
    end
end

@testset "MPI bcast" begin
    (MPI.Initialized() == false) && MPI.Init()
    comm = MPI.COMM_WORLD
    Nworker = MPI.Comm_size(comm)  # number of MPI workers
    rank = MPI.Comm_rank(comm)  # rank of current MPI worker
    root = 0 # rank of the root worker

    a = [1, 2, 3] .* rank 
    a = MCUtility.MPIbcast(a)
    if rank != root
        @test a == [0, 0, 0]
    end

    b = 1 .* rank
    b = MCUtility.MPIbcast(b)
    if rank != root
        @test b == 0
    end
end

@testset "MPI more" begin
    (MPI.Initialized() == false) && MPI.Init()
    comm = MPI.COMM_WORLD
    Nworker = MPI.Comm_size(comm)  # number of MPI workers
    rank = MPI.Comm_rank(comm)  # rank of current MPI worker
    root = 0 # rank of the root worker

    X = Continuous(0.0, 1.0)
    Y = Continuous(0.0, 1.0)
    Z = Continuous(0.0, 1.0)
    X.histogram[1] = 1.0
    Y.histogram[1] = 1.0
    Z.histogram[1] = 1.0
    cvar = CompositeVar(Y, Z)
    config = Configuration(var = (X, cvar), dof=[[1, 1], ])
    config.normalization = 1.0
    config.visited[1] = 1.0
    config.propose[1, 1, 1] = 1.0
    config.accept[1, 1, 1] = 1.0

    MCIntegration.MPIreduceConfig!(config)
    if rank == root
        @test config.normalization == Nworker
        @test config.visited[1] == Nworker
        @test config.propose[1, 1, 1] == Nworker
        @test config.accept[1, 1, 1] == Nworker
        @test config.var[1].histogram[1] == Nworker
        cvar = config.var[2]
        @test cvar[1].histogram[1] == Nworker
        @test cvar[2].histogram[1] == Nworker
    end

end