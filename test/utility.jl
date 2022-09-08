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

@testset "Configuration" begin
    # 3 integrals, 4 variables
    dof = [[1, 2, 3, 5], [3, 1, 2, 7], [2, 4, 1, 2]]
    @test MCIntegration._maxdof(dof) == [3, 4, 3, 7]
end
@testset "Probability" begin
    X = Continuous(0.0, 1.0; grid=[0.0, 0.1, 0.4, 1.0])
    Y = Continuous(0.0, 1.0; grid=[0.0, 0.2, 0.6, 1.0])
    Z = Discrete(1, 6; distribution=rand(6))

    config = Configuration(var=(X, Y, Z), dof=[[1, 1, 1], [2, 3, 3],])
    for vi in config.var
        Dist.initialize!(vi, config)
    end
    total_p = Dist.total_probability(config)
    for i in 1:config.N
        @test total_p ≈ Dist.probability(config, i) * Dist.padding_probability(config, i)
    end

    # make some shift! operation, then test again
    for (vi, v) in enumerate(config.var)
        for i in 1:config.maxdof[vi]
            Dist.shift!(v, i + v.offset, config)
        end
    end
    total_p = Dist.total_probability(config)
    for i in 1:config.N
        @test total_p ≈ Dist.probability(config, i) * Dist.padding_probability(config, i)
    end

end