# Convenience functions for working with MPI, adapted from DFTK.jl
"""
Number of processors used in MPI. Can be called without ensuring initialization.
"""
mpi_nprocs(comm=MPI.COMM_WORLD) = (MPI.Init(); MPI.Comm_size(comm))
mpi_master(comm=MPI.COMM_WORLD) = (MPI.Init(); MPI.Comm_rank(comm) == 0)
mpi_root(comm=MPI.COMM_WORLD) = 0
mpi_rank(comm=MPI.COMM_WORLD) = (MPI.Init(); MPI.Comm_rank(comm))

mpi_sum(arr, comm::MPI.Comm) = MPI.Allreduce(arr, +, comm)
mpi_sum!(arr, comm::MPI.Comm) = MPI.Allreduce!(arr, +, comm)
mpi_min(arr, comm::MPI.Comm) = MPI.Allreduce(arr, min, comm)
mpi_min!(arr, comm::MPI.Comm) = MPI.Allreduce!(arr, min, comm)
mpi_max(arr, comm::MPI.Comm) = MPI.Allreduce(arr, max, comm)
mpi_max!(arr, comm::MPI.Comm) = MPI.Allreduce!(arr, max, comm)
mpi_mean(arr, comm::MPI.Comm) = mpi_sum(arr, comm) ./ mpi_nprocs(comm)
mpi_mean!(arr, comm::MPI.Comm) = (mpi_sum!(arr, comm); arr ./= mpi_nprocs(comm))

"""
    function MPIreduce(data, op = MPI.SUM)

Reduce data from MPI workers to root with the operation `op`. `data` can be an array or a scalar.
The root node returns the reduced data with the operation `op`, and other nodes return their own data.
"""
function MPIreduce(data, op = MPI.SUM)
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
        d = MPI.Reduce([data, ], MPI.SUM, root, comm) # root node gets the sum of observables from all blocks
        return rank == root ? d[1] : data
        # return output
    end
end

"""
    function MPIreduce!(data::AbstractArray, op = MPI.SUM)

Reduce data from MPI workers to root with the operation `op`. `data` should be an array.
"""
function MPIreduce!(data::AbstractArray, op = MPI.SUM)
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
        result = MPI.bcast([data, ], root, comm) # root node gets the sum of observables from all blocks
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

function choose_parallel(parallel::Symbol)
    if parallel == :auto
        if Threads.nthreads() > 1
            return :thread  # use threads only if MPI is not available
        else
            return :nothread  # use threads only if MPI is not available
        end
    else
        return parallel
    end
end

# function check_parallel(parallel::Symbol)
#     if parallel == :mpi
#         # @assert mpi_nprocs() > 1 "MPI is not available"
#         # julia may start with -t N, but we don't want to use threads
#         # should work for MPI worker >=1
#         return
#     elseif parallel == :thread
#         @assert mpi_nprocs() == 1 "MPI and threads cannot be used together"
#     elseif parallel == :serial
#         @assert mpi_nprocs() == 1 "MPI should not be used for serial calculations"
#         # julia may start with -t N, but we don't want to use threads
#     else
#         error("Unknown parallelization mode: $(parallel)")
#     end
# end

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
    return mpi_master() && (Threads.threadid() == 1)
end

# function root(parallel::Symbol) # only one thread of each MPI worker is the root
#     if parallel == :mpi
#         return mpi_master()
#     elseif parallel == :thread
#         return 1
#     elseif parallel == :serial
#         return 1
#     else
#         error("Unknown parallelization mode: $(parallel)")
#     end
# end

# rank of the current worker for both MPI and threads
function rank(parallel::Symbol)
    #if thread is off, then nthreads must be one. Only mpi_rank contributes
    # mpi_rank() always start with 0
    return Threads.threadid()*(nthreads(parallel)-1)+mpi_rank()+1
end

# number of total workers for both MPI and threads
function nworker(parallel::Symbol)
    return mpi_nprocs()*nthreads(parallel)
end