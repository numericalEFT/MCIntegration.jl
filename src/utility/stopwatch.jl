"""
    StopWatch(start, interval, callback)

Initialize a stopwatch. 

# Arguments
- `start::Float64`: initial time (in seconds)
- `interval::Float64` : interval to click (in seconds)
- `callback` : callback function after each click (interval seconds)
"""
mutable struct StopWatch
    start::Float64
    interval::Float64
    f::Function
    StopWatch(_interval, callback) = new(time(), _interval, callback)
end

"""
    check(stopwatch, parameter...)

Check stopwatch. If it clicks, call the callback function with the unpacked parameter
"""
function check(watch::StopWatch, parameter...)
    now = time()
    if now - watch.start > watch.interval
        watch.f(parameter...)
        watch.start = now
    end
end