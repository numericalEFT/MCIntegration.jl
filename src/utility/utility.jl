"""
Utility data structures and functions
"""
module MCUtility
using Test
using ..MPI

include("stopwatch.jl")
export StopWatch, check

include("color.jl")
export black, red, green, yellow, blue, magenta, cyan, white

export progressBar, locate, smooth, rescale
"""
    progressBar(step, total)

Return string of progressBar (step/total*100%)
"""
function progressBar(step, total)
    barWidth = 70
    percent = round(step / total * 100.0, digits=2)
    str = "["
    pos = barWidth * percent / 100.0
    for i = 1:barWidth
        if i <= pos
            # str *= "█"
            str *= "="
        else
            str *= " "
        end
    end
    str *= "] $step/$total=$percent%"
    return str
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

function test_type_stability(f, args)
    try
        @inferred f(args...)
    catch e
        if isa(e, MethodError)
            @warn("call $f with wrong args. Got $(args)")
        else
            @warn "Type instability issue detected for $f, it may makes the integration slow" exception = (e, catch_backtrace())
            # @warn("Type instability issue detected for $f, it may makes the integration slow.\n$e")
        end
    end
end

end