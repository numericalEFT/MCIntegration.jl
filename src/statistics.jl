"""
    struct Result{O,C}

the returned result of the MC integration.

# Members

- `mean`: mean of the MC integration
- `stdev`: standard deviation of the MC integration
- `chi2`: chi-square of the MC integration
- `neval`: number of evaluations of the integrand
- `dof`: degrees of freedom of the MC integration (number of iterations - 1)
- `config`: configuration of the MC integration from the last iteration
- `iteractions`: list of tuples [(data, error, Configuration), ...] from each iteration
"""
struct Result{O,C}
    mean::O
    stdev::O
    chi2::O
    neval::Int
    dof::Int
    config::C
    iterations::Any
    function Result(history::AbstractVector)
        @assert length(history) > 0
        O = typeof(history[end][1])
        config = history[end][3]
        dof = length(history) - 1
        neval = sum(h[3].neval for h in history)
        mean, stdev, chi2 = average(history, dof + 1)
        # println(mean, ", ", stdev, ", ", chi2)
        return new{O,typeof(config)}(mean, stdev, chi2, neval, dof, config, history)
    end
end

function tostring(mval, merr; pm="±")
    # println(mval, ", ", merr)
    return @sprintf("%16.8g %s %.8g", mval, pm, merr)
    # val = if iszero(merr) || !isfinite(merr)
    #     mval
    # else
    #     err_digits = -Base.hidigit(merr, 10) + error_digits
    #     digits = if isfinite(mval)
    #         max(-Base.hidigit(mval, 10) + 2, err_digits)
    #     else
    #         err_digits
    #     end
    #     round(mval, digits=digits)
    # end
    # return "$val $pm $(round(merr, sigdigits=error_digits))"
end

"""
    function summary(result::Result, pick::Union{Function,AbstractVector}=obs -> real(first(obs)))

print the summary of the result. 
It will first print the configuration from the last iteration, then print the weighted average and standard deviation of the picked observable from each iteration.
The pick function is used to select one of the observable to be printed. The return value of pick function must be a Number.

"""
function summary(result::Result, pick::Union{Function,AbstractVector}=obs -> real(first(obs)))
    summary(result.config)

    if pick isa Function
        pick = [pick,]
    else
        @assert eltype(pick) <: Function "pick must be either a function or a vector of functions!"
    end
    for (i, p) in enumerate(pick)
        barbar = "==================================     Results#$i    =============================================="
        bar = "---------------------------------------------------------------------------------------------------"
        println(barbar)
        println(yellow(@sprintf("%6s %-36s %-36s %16s", "iter", "         integral", "        wgt average", "chi2/dof")))
        println(bar)
        for iter in 1:result.dof+1
            m0, e0 = p(result.iterations[iter][1]), p(result.iterations[iter][2])
            m, e, chi2 = average(result.iterations, iter)
            m, e, chi2 = p(m), p(e), p(chi2)
            println(@sprintf("%6s %-36s %-36s %16.4f", iter, tostring(m0, e0), tostring(m, e), iter == 1 ? 0.0 : chi2 / (iter - 1)))
        end
        println(bar)
        m, e = p(result.mean), p(result.stdev)
        println(green("result#$i = $m ± $e"))
        println()
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

"""

    function average(history, max=length(history))

average the history[1:max]. Return the mean, standard deviation and chi2 of the history.

# Arguments
- `history`: a list of tuples, such as [(data, error, Configuration), ...]
- `max`: the number of data to average over
"""
function average(history, max=length(history))
    @assert max > 0
    if max == 1
        return history[1][1], history[1][2], zero(history[1][1])
    end

    function _statistic(data, weight)
        @assert length(data) == length(weight)
        # println(data, " and ", weight)
        weightsum = sum(weight)
        mea = sum(data[i] .* weight[i] ./ weightsum for i in 1:max)
        err = 1.0 ./ sqrt.(weightsum)
        if max > 1
            chi2 = sum(weight[i] .* (data[i] - mea) .^ 2 for i in 1:max)
        else
            chi2 = zero(mea)
        end
        return mea, err, chi2
    end

    if eltype(history[end][1]) <: Complex
        dataR = [real.(history[i][1]) for i in 1:max]
        dataI = [imag.(history[i][1]) for i in 1:max]
        weightR = [1.0 ./ real.(history[i][2]) .^ 2 for i in 1:max]
        weightI = [1.0 ./ imag.(history[i][2]) .^ 2 for i in 1:max]
        mR, eR, chi2R = _statistic(dataR, weightR)
        mI, eI, chi2I = _statistic(dataI, weightI)
        return mR + mI * 1im, eR + eI * 1im, chi2R + chi2I * 1im
    else
        data = [history[i][1] for i in 1:max]
        weight = [1.0 ./ history[i][2] .^ 2 for i in 1:max]
        return _statistic(data, weight)
    end
end
