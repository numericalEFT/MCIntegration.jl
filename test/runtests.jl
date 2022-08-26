using MCIntegration
using Test

function check(result, expect)
    mean, error = result.mean, result.stdev
    for ei in eachindex(expect)
        @test abs(mean[ei] - expect[ei]) < 5.0 * error[ei]
    end
end

# @testset "MCIntegration.jl" begin
# Write your tests here.
include("utility.jl")
include("montecarlo.jl")
include("thread.jl")
# end
