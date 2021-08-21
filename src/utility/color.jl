# The most basic terminals have a set of 8 different colors

@inline function black(str)
    return "\u001b[30m$str\u001b[0m"
end

@inline function red(str)
    return "\u001b[31m$str\u001b[0m"
end

@inline function green(str)
    return "\u001b[32m$str\u001b[0m"
end

@inline function yellow(str)
    return "\u001b[33m$str\u001b[0m"
end

@inline function blue(str)
    return "\u001b[34m$str\u001b[0m"
end

@inline function magenta(str)
    return "\u001b[35m$str\u001b[0m"
end

@inline function cyan(str)
    return "\u001b[36m$str\u001b[0m"
end

@inline function white(str)
    return "\u001b[37m$str\u001b[0m"
end