"""
Utility data structures and functions
"""
module MCUtility
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
            str *= "█"
        else
            str *= " "
        end
    end
    str *= "] $step/$total=$percent%"
    return str
end

"""
    function locate(accumulation, p)
    
    Return index of p in accumulation so that accumulation[idx]<=p<accumulation[idx+1]. 
    If p is not in accumulation (namely accumulation[1] > p or accumulation[end] <= p), return -1.
    Bisection algorithmn is used so that the time complexity is O(log(n)) with n=length(accumulation).
"""
function locate(accumulation::AbstractVector, p::Number)
    n = length(accumulation)

    if accumulation[1] > p || accumulation[end] <= p
        error("$p is not in $accumulation")
        return -1
    end

    jl, ju = 1, n + 1
    while (ju - jl > 1)
        jm = (jl + ju) ÷ 2
        if p < accumulation[jm]
            ju = jm
        else
            jl = jm
        end
    end
    for i = 1:length(accumulation)
        if accumulation[i] > p
            @assert jl + 1 == i "$jl vs $i"
            return jl
        end
    end
    # return jl + 1
    # error("p=$p is out of the upper bound $(accumulation[end])")
end

"""
function smooth(dist::AbstractVector, factor=6)

    Smooth the distribution by averaging two nearest neighbor. The average ratio is given by 1 : factor : 1 for the elements which are not on the boundary.
"""
function smooth(dist::AbstractVector, factor=6)
    if length(dist) <= 1
        return dist
    end
    new = deepcopy(dist)
    new[1] = (dist[1] * (factor + 1) + dist[2]) / (factor + 2)
    new[end] = (dist[end] * (factor + 1) + dist[end-1]) / (factor + 2)
    for i = 2:length(dist)-1
        new[i] = (dist[i-1] + dist[i] * factor + dist[i+1]) / (factor + 2)
    end
    return new
end

"""
function rescale(dist::AbstractVector, alpha=1.5)

    rescale the dist array to avoid overreacting to atypically large number.
    There are three steps:
    1. dist will be first normalize to [0, 1].
    2. Then the values that are close to 1.0 will not be changed much, while that close to zero will be amplified to a value controlled by alpha.
    3. In the end, the rescaled dist array will be normalized to [0, 1].
    Check Eq. (19) of https://arxiv.org/pdf/2009.05112.pdf for more detail
"""
function rescale(dist::AbstractVector, alpha=1.5)
    if length(dist) == 1
        return dist
    end
    dist ./= sum(dist)
    @assert all(x -> (0 < x < 1), dist) "$dist"
    dist = @. ((1 - dist) / log(1 / dist))^alpha
    return dist ./= sum(dist)
end

end