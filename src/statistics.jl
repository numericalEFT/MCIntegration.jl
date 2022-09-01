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
    if isfinite(mval) && isfinite(merr)
        return @sprintf("%16.8g %s %.8g", mval, pm, merr)
    else
        return "$mval $pm $merr"
    end

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

function Base.show(io::IO, result::Result)
    # print(io, summary(result.config))
    print(io, report(result; verbose=-1))
end

"""
    function report(result::Result, pick::Union{Function,AbstractVector}=obs -> real(first(obs)), name=nothing)

print the summary of the result. 
It will first print the configuration from the last iteration, then print the weighted average and standard deviation of the picked observable from each iteration.

# Arguments
- result: Result object contains the history from each iteration
- pick: The pick function is used to select one of the observable to be printed. The return value of pick function must be a Number.
- name: name of each picked observable. If name is not given, the index of the pick function will be used.
"""
function report(result::Result, pick::Union{Function,AbstractVector}=obs -> real(first(obs)), name=nothing; verbose=0)
    # summary(result.config)

    # if pick isa Function
    #     pick = [pick,]
    # else
    #     @assert eltype(pick) <: Function "pick must be either a function or a vector of functions!"
    # end

    if isnothing(name) == false
        name = collect(name)
    end

    for i in eachindex(result.mean)
        # for (i, p) in enumerate(pick)
        p = pick
        info = isnothing(name) ? "$i" : "$(name[i])"
        if verbose >= 0
            barbar = "==================================     Integral $info    =============================================="
            bar = "---------------------------------------------------------------------------------------------------"
            println(barbar)
            println(yellow(@sprintf("%6s %-36s %-36s %16s", "iter", "         integral", "        wgt average", "chi2/dof")))
            println(bar)
            for iter in 1:result.dof+1
                m0, e0 = p(result.iterations[iter][1][i]), p(result.iterations[iter][2][i])
                m, e, chi2 = average(result.iterations, iter, i)
                m, e, chi2 = p(m[i]), p(e[i]), p(chi2[i])
                println(@sprintf("%6s %-36s %-36s %16.4f", iter, tostring(m0, e0), tostring(m, e), iter == 1 ? 0.0 : chi2 / (iter - 1)))
            end
            println(bar)
        else
            m, e, chi2 = p(result.mean[i]), p(result.stdev[i]), p(result.chi2[i])
            if result.dof == 0
                println(green("Integral $info = $m ± $e"))
            else
                println(green("Integral $info = $m ± $e   (chi2/dof = $(round(chi2/result.dof, sigdigits=3)))"))
            end
        end
        # println()
    end
    # end
end

"""

    function average(history, max=length(history))

average the history[1:max]. Return the mean, standard deviation and chi2 of the history.

# Arguments
- `history`: a list of tuples, such as [(data, error, Configuration), ...]
- `max`: the number of data to average over
"""
function average(history, max=length(history), idx=1)
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
        weightR = [1.0 ./ (real.(history[i][2]) .+ 1.0e-10) .^ 2 for i in 1:max]
        weightI = [1.0 ./ (imag.(history[i][2]) .+ 1.0e-10) .^ 2 for i in 1:max]
        mR, eR, chi2R = _statistic(dataR, weightR)
        mI, eI, chi2I = _statistic(dataI, weightI)
        return mR + mI * 1im, eR + eI * 1im, chi2R + chi2I * 1im
    else
        data = [history[i][1] for i in 1:max]
        weight = [1.0 ./ (history[i][2] .+ 1.0e-10) .^ 2 for i in 1:max]
        return _statistic(data, weight)
    end
end
