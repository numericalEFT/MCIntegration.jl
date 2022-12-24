"""
    struct Result{O,C}

the returned result of the MC integration.

# Members

- `mean`: mean of the MC integration
- `stdev`: standard deviation of the MC integration
- `chi2`: chi-square per dof of the MC integration
- `neval`: number of evaluations of the integrand
- `ignore`: ignore iterations untill `ignore`
- `dof`: degrees of freedom of the MC integration (number of iterations - 1)
- `config`: configuration of the MC integration from the last iteration
- `iterations`: list of tuples [(data, error, Configuration), ...] from each iteration
"""
struct Result{O,C}
    mean::O
    stdev::O
    chi2::Any
    neval::Int
    ignore::Int # ignore iterations untill ignore_iter
    dof::Int
    config::C
    iterations::Any
    function Result(history::AbstractVector, ignore::Int)
        # history[end][1] # a vector of avg
        # history[end][2] # a vector of std
        # history[end][3] # a vector of config
        init = ignore + 1
        @assert length(history) > 0
        config = history[end][3]
        dof = (length(history) - init + 1) - 1 # number of effective samples - 1
        neval = sum(h[3].neval for h in history)
        @assert config.N >= 1
        if config.N == 1
            O = typeof(history[end][1][1]) #if there is only value, then extract this value from the vector
            mean, stdev, chi2 = average(history, 1; init=init, max=length(history))
        else
            O = typeof(history[end][1])
            @assert O <: AbstractVector
            mean, stdev, chi2 = [], [], []
            res = [average(history, o; init=init, max=length(history)) for o in 1:config.N]
            mean = [r[1] for r in res]
            stdev = [r[2] for r in res]
            chi2 = [r[3] for r in res]
            # for o in 1:config.N
            #     _mean, _stdev, _chi2 = average(history, dof + 1, o)
            #     push!(mean, _mean)
            #     push!(stdev, _stdev)
            #     push!(chi2, _chi2)
            # end
        end
        # println(mean, ", ", stdev, ", ", chi2)
        # println(typeof(mean), typeof(config))
        return new{O,typeof(config)}(mean, stdev, chi2, neval, ignore, dof, config, history)
    end
    function Result(res::Result, ignore::Int)
        if ignore == res.ignore
            return res
        else
            return Result(res.iterations, ignore)
        end
    end
end

function Base.getindex(result::Result, idx::Int)
    return result.mean[idx], result.stdev[idx], result.chi2[idx]
end

"""
Convert a measurement to its string representation.

- sigdigits: Sets the number of significant figures in the measurement value.
             If unspecified (or negative), the entire measurement value is regarded as significant.
             The error bar will be rounded to the nearest significant decimal place.

- fallback_mode: Sets the fallback behavior when the error bar is less than
                 the least significant order of magnitude of the measurement.
                 If `parenthetical`, the measurement is serialized as `val(err)`.
                 If `plusminus`, it is serialized as `val ± err`.
"""
function stringrep(meas; sigdigits=-1, fallback_mode="plusminus")
    # Floating-point values will be preserved to at least 10ϵ in the following manipulations
    fp_sigdigits = 15

    # Round value to specified digits if applicable
    if sigdigits ≥ 0
        val = round(meas.val, sigdigits=sigdigits)
    else
        val = round(meas.val, digits=fp_sigdigits)
    end
    err = round(meas.err, digits=fp_sigdigits)

    # No error bar
    if err == 0
        return "$val"
    end

    # Base 10 exponents
    val_exp = Int(floor(log10(abs(val))))
    err_exp = Int(floor(log10(abs(err))))

    # Base 10 value significand
    val_signif = round(val / 10.0^val_exp, digits=fp_sigdigits)
    err_signif = round(err / 10.0^err_exp, digits=fp_sigdigits)

    # Error rescaled to value order of magnitude (OOM).
    err_rescaled = round(err / 10.0^val_exp, digits=fp_sigdigits)

    # Are the value/error reported in scientific notation?
    val_in_scinotn = occursin("e", "$val")
    err_in_scinotn = occursin("e", "$err")

    # Regex matching the exponent of a digit in scientific notation
    regex_exp = r"e[+-]?\d+"

    # Number of displayed decimal places in the measurement value
    val_split = split(replace("$val", regex_exp => ""), ".")
    err_split = split(replace("$err", regex_exp => ""), ".")
    ndec_val = length(val_split) > 1 ? length(val_split[end]) : 0
    ndec_err = length(err_split) > 1 ? length(err_split[end]) : 0
    ndec_err_rescaled = max(0, ndec_err - val_exp)
    ndigits_val = sum(length.(val_split))

    # Fallback mode: Error is smaller OOM than least significant value digit.
    #                E.x.: (v = 1.3e-6, e = 1.002e-9), (v = 1.302e9, e = 1.002e4)
    # exp_diff = val_exp - err_exp
    # ls_oom_diff = exp_diff - ndec_val

    # Left and right padding mismatches between value significand and rescaled error
    left_mismatch = val_exp - err_exp  # +ve: no fallback, -ve: fallback
    right_mismatch = ndec_val - ndec_err   # +ve: rpad e, -ve: rpad v

    # Pad value with trailing zeros when needed to match the requested precision
    val_pad = ""
    if sigdigits > ndigits_val
        val_pad *= repeat("0", sigdigits - ndigits_val)
    end

    # fallback = abs(err_exp) < abs(val_exp) - ndec_val

    # An error bar with larger order of magnitude (OOM) than val requires zero-padding
    # err_pad = ndec_val - ndec_err + err_exp
    # err_pad = max(0, exp_diff + ndec_val)
    # err_pad = max(0, err_exp - val_exp)

    # Fall back to an explicit notation if the error bar is too small
    fallback = ndec_val - left_mismatch < 0
    fallback = false
    if fallback && fallback_mode != "padvalue"
        println("Using fallback mode '$(fallback_mode)'...")
        rerr = round(err, sigdigits=fp_sigdigits)
        if fallback_mode == "parenthetical"
            meas_string = "$val($rerr)"
        elseif fallback_mode == "plusminus"
            meas_string = "$val ± $rerr"
        else
            error("Fallback mode should be either 'parenthetical' or 'plusminus'.")
        end
        # Summarize results
        println("Measurement string: $meas_string")
        println("Original measurement: $meas")
        # NOTE: Parenthetical fallback mode with scientific notation is not supported by the
        #       measurement parser, but offers a more compact version of the plus/minus notation.
        if fallback_mode == "parenthetical" && fallback && val_in_scinotn
            parsable_meas_string = "$val ± $rerr"
            println("Resulting measurement: $(measurement(parsable_meas_string))")
        else
            println("Resulting measurement: $(measurement(meas_string))")
        end
        # @assert measurement(meas_string) ≈ meas
        return meas_string
    end

    # Error integer representation before possible rounding/padding.
    # Any leading zeros are removed by parsing the string as an Int.
    # err_int = parse(Int, replace("$err", "." => "", regex_exp => ""))

    if right_mismatch > 0
        # Integer significand for error at OOM set by value exponent
        # err_int = parse(Int, split("$(err / 10.0^val_exp)", ".")[end])

        # err_sigdigits = 1 - ls_oom_diff

        # Crop error integer to match value decimals or pad with zeros if needed
        # if err_sigdigits < length(err_sint)
        #     err_string = "$err_sint"[1:right_mismatch]
        # else
        #     err_string = "$err_sint" * repeat("0", right_mismatch)
        # end

        # Error padding required
        err_int = parse(Int, replace("$err", "." => "", regex_exp => ""))
        err_string = "$err_int" * repeat("0", right_mismatch)
        meas_string = "$val_signif($err_string)"
    else
        # Value padding (if applicable) and/or error rounding required
        if fallback_mode == "padvalue"
            "Using fallback mode 'padvalue'..."
            val_pad += repeat("0", -right_mismatch)
            meas_string = "$(val_signif)$(val_pad)($err_int)"
        else
            # sigdigits = min(fp_sigdigits, length("$err_int") + right_mismatch)
            # rounded_err_int = Int(round(err_int, sigdigits=sigdigits))
            # rounded_err_int = Int(round(err_int, sigdigits=ndigits_val))
            # rounded_err_string = "$rounded_err_int"[1:sigdigits]
            # meas_string = "$val_signif($(rounded_err_string))"

            rounded_err = round(err_rescaled, digits=ndec_val + length(val_pad))
            if rounded_err == 0
                meas_string = "$(val_signif)$(val_pad)"
            else
                rounded_err_string = replace("$rounded_err", "." => "")
                meas_string = "$(val_signif)$(val_pad)($(rounded_err_string))"
                # meas_string = "$val_signif($(rounded_err_int))"
            end
        end
        # err_string = "$rerr_int"[1:err_sigdigits] * repeat("0", right_mismatch)
    end

    if val_in_scinotn
        meas_string *= "e$(val_exp)"
    end

    # Summarize results
    println("Measurement string: $meas_string")
    println("Original measurement: $meas")
    # NOTE: Parenthetical fallback mode with scientific notation is not supported by the
    #       measurement parser, but offers a more compact version of the plus/minus notation.
    if fallback_mode == "parenthetical" && fallback && val_in_scinotn
        parsable_meas_string = "$val ± $rerr"
        println("Resulting measurement: $(measurement(parsable_meas_string))\n")
    else
        println("Resulting measurement: $(measurement(meas_string))\n")
    end
    # @assert measurement(meas_string) ≈ meas
    return meas_string
end

function tostring(mval, merr; pm="±")
    # println(mval, ", ", merr)

    if mval isa Real && merr isa Real && isfinite(mval) && isfinite(merr)
        m = @sprintf("%16.8g %s %-16.8g", mval, pm, merr)
        # m = measurement(mval, merr)
        # return @sprintf("$m")
    elseif mval isa Complex && merr isa Complex && isfinite(mval) && isfinite(merr)
        # Padded measurement string without parenthesis or 'im' token
        str = @sprintf("%16.6g %s %16.6g + %16.6g %s %16.6g", real(mval), pm, real(merr), imag(mval), pm, imag(merr))
        # Match each measurement value plus padding to be replaced with tokens
        padded_num_rx = r"(\s?)(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?)(\s{0,3})"
        tokens = ["(", ")  ", "(", ")im"]
        sides = ["left", "right", "left", "right"]  # Replace padding to the left or right?
        # Insert parenthesis and 'im' tokens into measurement string
        m = str
        for (i, padded_num) in enumerate(eachmatch(padded_num_rx, str))
            if sides[i] == "left"
                # Add prefix token to left padding
                patch = padded_num.match => "$(tokens[i])$(padded_num[2])$(padded_num[3])"
            else
                # Add suffix token to right padding
                patch = padded_num.match => "$(padded_num[1])$(padded_num[2])$(tokens[i])"
            end
            m = replace(m, patch, count=1)
        end
        m = @sprintf("%8.6g(%3g) + %-8.6g(%3g)im", real(mval), real(merr), imag(mval), imag(merr))
        # m = measurement(real(mval), real(merr)) + measurement(imag(mval), imag(merr)) * 1im
        # return @sprintf("%16.6g(%6g) + %16.6g(%6g)im", real(mval), real(merr), imag(mval), imag(merr))
    else
        m = "$mval $pm $merr"
    end

    return "$m"
end

function Base.show(io::IO, result::Result)
    # print(io, summary(result.config))
    # print(io, report(result; verbose=-1, io = io))
    for i in 1:result.config.N
        info = "$i"
        m, e, chi2 = first(result.mean[i]), first(result.stdev[i]), first(result.chi2[i])
        if result.dof == 0
            print(io, green("Integral $info = $m ± $e"))
        else
            print(io, green("Integral $info = $m ± $e   (chi2/dof = $(round(chi2, sigdigits=3)))"))
        end
        if i < result.config.N
            print(io, "\n")
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", result::Result)
    Base.show(io, result)
end

"""
    function report(result::Result, ignore=result.ignore; pick::Union{Function,AbstractVector}=obs -> first(obs), name=nothing, verbose=0)

print the summary of the result. 
It will first print the configuration from the last iteration, then print the weighted average and standard deviation of the picked observable from each iteration.

# Arguments
- result: Result object contains the history from each iteration
- ignore: the ignore the first # iteractions.
- pick: The pick function is used to select one of the observable to be printed. The return value of pick function must be a Number.
- name: name of each picked observable. If name is not given, the index of the pick function will be used.
"""
function report(result::Result, ignore=result.ignore; pick::Union{Function,AbstractVector}=obs -> first(obs), name=nothing, verbose=0, io::IO=Base.stdout)
    if isnothing(name) == false
        name = collect(name)
    end
    ignore_iter = ignore

    for i in 1:result.config.N
        p = pick
        info = isnothing(name) ? "$i" : "$(name[i])"
        if verbose >= 0
            # barbar = "==============================================     Integral $info    =========================================================="
            # bar = "---------------------------------------------------------------------------------------------------------------------------"
            barbar = "====================================     Integral $info    ================================================"
            bar = "-------------------------------------------------------------------------------------------------------"
            println(io, barbar)
            println(io, yellow(@sprintf("%6s     %-32s     %-32s %16s", "iter", "         integral", "        wgt average", "chi2/dof")))
            println(io, bar)
            for iter in 1:length(result.iterations)
                m0, e0 = p(result.iterations[iter][1][i]), p(result.iterations[iter][2][i])
                m, e, chi2 = average(result.iterations, i; init=ignore_iter + 1, max=iter)
                m, e, chi2 = p(m), p(e), p(chi2)
                iterstr = iter <= ignore_iter ? "ignore" : "$iter"
                sm0, sm = tostring(m0, e0), tostring(m, e)
                println(io, @sprintf("%6s %36s %36s %16.4f", iterstr, sm0, sm, abs(chi2)))
            end
            println(io, bar)
        else
            m, e, chi2 = p(result.mean[i]), p(result.stdev[i]), p(result.chi2[i])
            if result.dof == 0
                println(io, green("Integral $info = $m ± $e"))
            else
                println(io, green("Integral $info = $m ± $e   (chi2/dof = $(round(chi2, sigdigits=3)))"))
            end
        end
    end
end

"""

    function average(history, idx=1; init=1, max=length(history))

average the history[1:max]. Return the mean, standard deviation and chi2 of the history.

# Arguments
- `history`: a list of tuples, such as [(data, error, Configuration), ...]
- `idx`: the index of the integral
- `max`: the last index of the history to average with
- `init` : the first index of the history to average with
"""
function average(history, idx=1; init=1, max=length(history))
    @assert max > 0
    @assert init > 0
    if max <= init
        return history[1][1][idx], history[1][2][idx], zero(history[1][1][idx])
    end

    function _statistic(data, weight)
        @assert length(data) == length(weight)
        # println(data, " and ", weight)
        weightsum = sum(weight)
        mea = sum(data[i] .* weight[i] ./ weightsum for i in eachindex(weight))
        err = 1.0 ./ sqrt.(weightsum)
        if max > 1
            chi2 = sum(weight[i] .* (data[i] - mea) .^ 2 for i in eachindex(weight))
        else
            chi2 = zero(mea)
        end
        return mea, err, chi2 / ((max - init + 1) - 1)
    end

    if eltype(history[end][1][idx]) <: Complex
        dataR = [real.(history[i][1][idx]) for i in init:max]
        dataI = [imag.(history[i][1][idx]) for i in init:max]
        weightR = [1.0 ./ (real.(history[i][2][idx]) .+ 1.0e-10) .^ 2 for i in init:max]
        weightI = [1.0 ./ (imag.(history[i][2][idx]) .+ 1.0e-10) .^ 2 for i in init:max]
        mR, eR, chi2R = _statistic(dataR, weightR)
        mI, eI, chi2I = _statistic(dataI, weightI)
        return mR + mI * 1im, eR + eI * 1im, chi2R + chi2I * 1im
    else
        data = [history[i][1][idx] for i in init:max]
        weight = [1.0 ./ (history[i][2][idx] .+ 1.0e-10) .^ 2 for i in init:max]
        return _statistic(data, weight)
    end
end