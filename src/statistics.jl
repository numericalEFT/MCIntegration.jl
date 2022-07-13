
function reduceConfig(c::Configuration, root, comm)
    if MPI.Comm_rank(comm) == root
        # reweight ./= MPI.Comm_size(comm)
        # return SummaryStat(neval, visited, reweight, propose, accept)
        rc = deepcopy(c)
        rc.neval = MPI.Reduce(c.neval, MPI.SUM, root, comm)
        rc.visited = MPI.Reduce(c.visited, MPI.SUM, root, comm)
        rc.propose = MPI.Reduce(c.propose, MPI.SUM, root, comm)
        rc.accept = MPI.Reduce(c.accept, MPI.SUM, root, comm)
        rc.observable = MPI.Reduce(c.observable, MPI.SUM, root, comm)
        rc.normalization = MPI.Reduce(c.normalization, MPI.SUM, root, comm)
        return rc
    else
        MPI.Reduce(c.neval, MPI.SUM, root, comm)
        MPI.Reduce(c.visited, MPI.SUM, root, comm)
        MPI.Reduce(c.propose, MPI.SUM, root, comm)
        MPI.Reduce(c.accept, MPI.SUM, root, comm)
        MPI.Reduce(c.observable, MPI.SUM, root, comm)
        MPI.Reduce(c.normalization, MPI.SUM, root, comm)
        return c
    end
end

function MPIreduce(data)
    comm = MPI.COMM_WORLD
    Nworker = MPI.Comm_size(comm)  # number of MPI workers
    rank = MPI.Comm_rank(comm)  # rank of current MPI worker
    root = 0 # rank of the root worker

    if Nworker == 1 #no parallelization
        return data
    end
    if typeof(data) <: AbstractArray
        MPI.Reduce!(data, MPI.SUM, root, comm) # root node gets the sum of observables from all blocks
        return data
    else
        result = [data,]  # MPI.Reduce works for array only
        MPI.Reduce!(result, MPI.SUM, root, comm) # root node gets the sum of observables from all blocks
        return result[1]
    end
end