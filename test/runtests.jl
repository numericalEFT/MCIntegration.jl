using MCIntegration
using Test

function check(mean, error, expect)
    # println(mean, error)
    for ei in eachindex(expect)
        @test abs(mean[ei] - expect[ei]) < 5.0 * error[ei]
    end
end

function check(result, expect)
    mean, error = result.mean, result.stdev
    check(mean, error, expect)
end

function check_complex(result, expect)
    mean, error = result.mean, result.stdev
    # println(mean, error)
    check(real(mean), real(error), real(expect))
    check(imag(mean), imag(error), imag(expect))
end

# @testset "MCIntegration.jl" begin
# Write your tests here.
include("utility.jl")
include("montecarlo.jl")
include("thread.jl")
# end
