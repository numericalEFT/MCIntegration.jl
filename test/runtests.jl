using MCIntegration
using Test

function check(mean, error, expect, ratio=7.0)
    println("mean = $mean, error = $error with expected value = $expect")
    for ei in eachindex(expect)
        @test abs(mean[ei] - expect[ei]) < error[ei] * ratio
    end
end

function check(result::Result, expect, ratio=7.0)
    # println(result)
    mean, error = result.mean, result.stdev
    check(mean, error, expect, ratio)
end

function check_vector(result::Result, expect, ratio=7.0)
    mean, error = result.mean, result.stdev
    for ei in eachindex(expect)
        check(mean[ei], error[ei], expect[ei], ratio)
    end
end

function check_complex(result::Result, expect, ratio=7.0)
    mean, error = result.mean, result.stdev
    # println(mean, error)
    check(real(mean), real(error), real(expect), ratio)
    check(imag(mean), imag(error), imag(expect), ratio)
end

# @testset "MCIntegration.jl" begin
# Write your tests here.
if isempty(ARGS)
    include("utility.jl")
    include("variable.jl")
    include("statistics.jl")
    include("montecarlo.jl")
    include("thread.jl")
    include("bubble.jl")
    include("bubble_FermiK.jl")
    include("mpi.jl")
    include("interface_tests.jl")
else
    include(ARGS[1])
end
# end
