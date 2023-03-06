# Convenience functions for working with MPI, adapted from DFTK.jl
"""
Number of processors used in MPI. Can be called without ensuring initialization.
"""
function mpi_nprocs(comm=MPI.COMM_WORLD)
    (MPI.Initialized() == false) && MPI.Init()
    MPI.Comm_size(comm)
end
function mpi_master(comm=MPI.COMM_WORLD)
    (MPI.Initialized() == false) && MPI.Init()
    MPI.Comm_rank(comm) == 0
end
function mpi_rank(comm=MPI.COMM_WORLD)
    (MPI.Initialized() == false) && MPI.Init()
    MPI.Comm_rank(comm)
end
mpi_root(comm=MPI.COMM_WORLD) = 0

"""
    function MPIreduce(data, op = MPI.SUM)

Reduce data from MPI workers to root with the operation `op`. `data` can be an array or a scalar.
The root node returns the reduced data with the operation `op`, and other nodes return their own data.
"""
function MPIreduce(data, op=MPI.SUM)
    comm = MPI.COMM_WORLD
    Nworker = MPI.Comm_size(comm)  # number of MPI workers
    rank = MPI.Comm_rank(comm)  # rank of current MPI worker
    root = 0 # rank of the root worker

    if Nworker == 1 #no parallelization
        return data
    end
    if typeof(data) <: AbstractArray
        d = MPI.Reduce(data, MPI.SUM, root, comm) # root node gets the sum of observables from all blocks
        return rank == root ? d : data
    else
        # MPI.Reduce works for array only in old verison of MPI.jl
        d = MPI.Reduce([data,], MPI.SUM, root, comm) # root node gets the sum of observables from all blocks
        return rank == root ? d[1] : data
        # return output
    end
end

"""
    function MPIreduce!(data::AbstractArray, op = MPI.SUM)

Reduce data from MPI workers to root with the operation `op`. `data` should be an array.
"""
function MPIreduce!(data::AbstractArray, op=MPI.SUM)
    comm = MPI.COMM_WORLD
    Nworker = MPI.Comm_size(comm)  # number of MPI workers
    rank = MPI.Comm_rank(comm)  # rank of current MPI worker
    root = 0 # rank of the root worker

    if Nworker > 1
        MPI.Reduce!(data, MPI.SUM, root, comm) # root node gets the sum of observables from all blocks
    end
end

"""
    function MPIbcast(data)

Broadcast data from MPI root to other nodes. `data` can be an array or a scalar.
The root node its own data, and other nodes return the broadcasted data from the root.
"""
function MPIbcast(data)
    comm = MPI.COMM_WORLD
    Nworker = MPI.Comm_size(comm)  # number of MPI workers
    rank = MPI.Comm_rank(comm)  # rank of current MPI worker
    root = 0 # rank of the root worker

    if Nworker == 1 #no parallelization
        return data
    end
    if typeof(data) <: AbstractArray
        return MPI.bcast(data, root, comm) # root node gets the sum of observables from all blocks
    else
        # MPI.Reduce works for array only
        result = MPI.bcast([data,], root, comm) # root node gets the sum of observables from all blocks
        return result[1]
    end
end

"""
    function MPIbcast!(data::AbstractArray)

Broadcast data from MPI root to other nodes. `data` is an array.
"""
function MPIbcast!(data::AbstractArray)
    comm = MPI.COMM_WORLD
    Nworker = MPI.Comm_size(comm)  # number of MPI workers
    rank = MPI.Comm_rank(comm)  # rank of current MPI worker
    root = 0 # rank of the root worker

    if Nworker > 1 #no parallelization
        return MPI.Bcast!(data, root, comm) # root node gets the sum of observables from all blocks
    end
end

# actual number of threads used
function nthreads(parallel::Symbol)
    if parallel == :thread
        return Threads.nthreads()
    else
        return 1
    end
end

# only one thread of each MPI worker is the root
function is_root(parallel::Symbol)
    if parallel == :thread
        return mpi_master() && (Threads.threadid() == 1)
    else
        return mpi_master()
    end
end

# # rank of the current worker for both MPI and threads
# function rank(parallel::Symbol)
#     #if thread is off, then nthreads must be one. Only mpi_rank contributes
#     # mpi_rank() always start with 0
#     if parallel == :thread
#         return Threads.threadid()+((nthreads(parallel)-1)*mpi_rank())
#     else
#         return mpi_rank()+1
#     end
# end

function threadid(parallel::Symbol)
    if parallel == :thread
        return Threads.threadid()
    else
        return 1
    end
end

# number of total workers for both MPI and threads
function nworker(parallel::Symbol)
    return mpi_nprocs() * nthreads(parallel)
end