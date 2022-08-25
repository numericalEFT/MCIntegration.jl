using MCIntegration
using Test

function check(result, expect)
    mean, error = result.mean, result.stdev
    @test abs(mean - expect) < 5.0 * error
end

function check(mean, error, expect)
    @test abs(mean - expect) < 5.0 * error
end

# @testset "MCIntegration.jl" begin
# Write your tests here.
include("utility.jl")
include("montecarlo.jl")
include("thread.jl")
# end
