module MCIntegrationMPIExt

using MCIntegration: integrate, ParallelBackend

function MCIntegrators.integrate(f, x, config::MPIBackend)
end

function doReweightMPI!(config::Configuration, gamma, reweight_goal::Union{Vector{Float64},Nothing}, comm::MPI.Comm)
    if MCUtility.mpi_master()
        # only the master process will output results, no matter parallel = :mpi or :thread or :serial
        doReweight!(config, gamma, reweight_goal)
    end
    reweight_array = Vector{Float64}(config.reweight)
    MPI.Bcast!(reweight_array, 0, comm)
    config.reweight .= reweight_array
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

end
