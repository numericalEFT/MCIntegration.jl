@testset "Utility" begin
    grid = [0.0, 0.1, 0.3, 0.5]
    @test Dist.locate(grid, eps(1.0)) == 1
    @test Dist.locate(grid, 0.5 - eps(1.0)) == length(grid) - 1
    @test Dist.locate(grid, grid[1]) == 1
    # @test MCIntegration.locate(grid, grid[end]) == length(grid)
    @test Dist.locate(grid, 0.05) == 1
    @test Dist.locate(grid, 0.2) == 2
    @test Dist.locate(grid, 0.31) == 3
end

@testset "Other" begin
    # 3 integrals, 4 variables
    dof = [[1, 2, 3, 5], [3, 1, 2, 7], [2, 4, 1, 2]]
    MCIntegration._maxdof(dof) == [3, 4, 3, 7]
end