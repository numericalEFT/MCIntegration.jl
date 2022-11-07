"""
Utility data structures and functions
"""
module MCUtility
using Test
using ..MPI
using ..Threads

include("stopwatch.jl")
export StopWatch, check

include("color.jl")
export black, red, green, yellow, blue, magenta, cyan, white

include("parallel.jl")

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