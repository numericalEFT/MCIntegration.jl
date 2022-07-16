@testset "Utility" begin
    grid = [0.0, 0.1, 0.3, 0.5]
    @test MCIntegration.locate(grid, eps(1.0)) == 1
    @test MCIntegration.locate(grid, 0.5 - eps(1.0)) == length(grid) - 1
    @test MCIntegration.locate(grid, grid[1]) == 1
    # @test MCIntegration.locate(grid, grid[end]) == length(grid)
    @test MCIntegration.locate(grid, 0.05) == 1
    @test MCIntegration.locate(grid, 0.2) == 2
    @test MCIntegration.locate(grid, 0.31) == 3
end