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
    aa = MCUtility.MPIreduce(a)
    if rank == root
        @test aa == [Nworker, 2Nworker, 3Nworker]
        @test a == [1, 2, 3]
    else
        @test aa == a #non-root returns nothing
    end

    b = 1
    bb = MCUtility.MPIreduce(b)
    if rank == root
        @test b == 1
        @test bb == Nworker
    else
        @test bb == b #non-root returns nothing
    end

    # inplace
    a = [1, 2, 3] 
    MCUtility.MPIreduce!(a)
    if rank == root
        @test a == [Nworker, 2Nworker, 3Nworker]
    end
end

@testset "MPI bcast" begin
    (MPI.Initialized() == false) && MPI.Init()
    comm = MPI.COMM_WORLD
    Nworker = MPI.Comm_size(comm)  # number of MPI workers
    rank = MPI.Comm_rank(comm)  # rank of current MPI worker
    root = 0 # rank of the root worker

    a = [1, 2, 3] .* rank 
    aa = MCUtility.MPIbcast(a)
    if rank != root
        @test aa == [0, 0, 0]
        @test a == [1, 2, 3] .* rank
    else
        @test a == aa
    end

    b = rank
    bb = MCUtility.MPIbcast(b)
    if rank != root
        @test bb == 0
        @test b == rank
    else
        @test b == bb
    end

    # inplace
    a = [1, 2, 3] .* rank 
    MCUtility.MPIbcast!(a)
    if rank != root
        @test a == [0, 0, 0]
    end

end

@testset "MPI reduce config" begin
    (MPI.Initialized() == false) && MPI.Init()
    comm = MPI.COMM_WORLD
    Nworker = MPI.Comm_size(comm)  # number of MPI workers
    rank = MPI.Comm_rank(comm)  # rank of current MPI worker
    root = 0 # rank of the root worker

    X = Continuous(0.0, 1.0)
    Y = Continuous(0.0, 1.0)
    Z = Continuous(0.0, 1.0)
    X.histogram[1] = 1.1
    Y.histogram[1] = 1.2
    Z.histogram[1] = 1.3
    cvar = CompositeVar(Y, Z)
    obs = [1.0,]
    config = Configuration(var = (X, cvar), dof=[[1, 1], ], obs=obs)
    config.neval = 101
    config.normalization = 1.1
    config.visited[1] = 1.2
    config.propose[1, 1, 1] = 1.3
    config.accept[1, 1, 1] = 1.4

    MCIntegration.MPIreduceConfig!(config)
    if rank == root
        @test config.observable[1] == Nworker
        @test config.neval == Nworker*101
        @test config.normalization ≈ Nworker*1.1
        @test config.visited[1] ≈ Nworker*1.2
        @test config.propose[1, 1, 1] ≈ Nworker*1.3
        @test config.accept[1, 1, 1] ≈ Nworker*1.4

        @test config.var[1].histogram[1] ≈ Nworker*1.1 # X
        cvar = config.var[2] #compositevar
        @test cvar[1].histogram[1] ≈ Nworker *1.2 #Y
        @test cvar[2].histogram[1] ≈ Nworker*1.3 #Z
    end
end

@testset "MPI bcast histogram" begin
    (MPI.Initialized() == false) && MPI.Init()
    comm = MPI.COMM_WORLD
    Nworker = MPI.Comm_size(comm)  # number of MPI workers
    rank = MPI.Comm_rank(comm)  # rank of current MPI worker
    root = 0 # rank of the root worker

    X = Continuous(0.0, 1.0)
    Y = Continuous(0.0, 1.0)
    Z = Continuous(0.0, 1.0)
    cvar = CompositeVar(Y, Z)

    if rank == root
        X.histogram[1] = 1.1
        Y.histogram[1] = 1.2
        Z.histogram[1] = 1.3
    else
        X.histogram[1] = rank
        Y.histogram[1] = rank
        Z.histogram[1] = rank
    end

    config = Configuration(var = (X, cvar), dof=[[1, 1], ])
    config.reweight = [1.1, 1.2]

    MCIntegration.MPIbcastConfig!(config)

    if rank != root
        @test config.reweight ≈ [1.1, 1.2]

        @test config.var[1].histogram[1] ≈ 1.1 # X
        cvar = config.var[2] #compositevar
        @test cvar[1].histogram[1] ≈ 1.2 #Y
        @test cvar[2].histogram[1] ≈ 1.3 #Z
    end
end