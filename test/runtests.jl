using MCIntegration
using Test

function check(mean, error, expect)
    println("check")
    @test abs(mean - expect) < 5.0 * error
end

# @testset "MCIntegration.jl" begin
# Write your tests here.
include("utility.jl")
include("montecarlo.jl")
include("thread.jl")
# end
