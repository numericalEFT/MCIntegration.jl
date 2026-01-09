using MPI
using Test

@testset "MPI" begin
    n = 2  # number of processes
    mpiexec() do exe  # MPI wrapper
        run(`$exe -n $n $(Base.julia_cmd()) mpi_test.jl`)
        # alternatively:
        # p = run(ignorestatus(`...`))
        # @test success(p)
    end
end