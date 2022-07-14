"""
Utility data structures and functions
"""
module MCUtility
include("stopwatch.jl")
export StopWatch, check

include("color.jl")
export black, red, green, yellow, blue, magenta, cyan, white

export progressBar, locate
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
            str *= "█"
        else
            str *= " "
        end
    end
    str *= "] $step/$total=$percent%"
    return str
end

function locate(accumulation, p)
    for i = 1:length(accumulation)
        if accumulation[i] > p
            return i
        end
    end
    error("p=$p is out of the upper bound $(accumulation[end])")
end

end