

































using .Measurements

@testset verbose = true "Measurement Serializer" begin
    # Test 1: err > val, N > 0
    s1 = "1.002(345600)"
    m1 = measurement(s1)
    println("Test case: $s1")
    @test MCIntegration.stringrep(m1) == s1

    # Test 2: err < val, trailing zeros => sigdigits = 9
    s2 = "100000.100(10001)"
    m2 = measurement(s2)
    println("Test case: $s2")
    @test MCIntegration.stringrep(m2) == s2
    # @test MCIntegration.stringrep(m2; sigdigits=9) == s2
    # Test 3: Scientific notation, err > val, N > 0
    s3 = "1.209821(123098098)e6"
    m3 = measurement(s3)
    println("Test case: $s3")
    @test MCIntegration.stringrep(m3) == s3

    # Test 4: Scientific notation, err > val, N < 0
    s4 = "1.0432(100234099)e-10"
    m4 = measurement(s4)
    println("Test case: $s4")
    @test MCIntegration.stringrep(m4) == s4

    # Test 5: Scientific notation, err < val, N > 0
    s5 = "1.23456(1000)e4"
    m5 = measurement(s5)
    println("Test case: $s5")
    @test MCIntegration.stringrep(m5) == s5
    # Test 6: Scientific notation, err < val, N < 0,
    #         trailing zeros => sigdigits = 7
    s6 = "1.300000(100234099)e-6"
    m6 = measurement(s6)
    println("Test case: $s6")
    @test MCIntegration.stringrep(m6; sigdigits=7) == s6

    # Test 7: No error bar
    s7 = "1.3000001"
    m7 = measurement(s7)
    println("Test case: $s7")
    @test MCIntegration.stringrep(m7) == s7
    # Test 8: Scientific notation, no error bar
    s8 = "5.50144e-6"
    m8 = measurement(s8)
    println("Test case: $s8")
    @test MCIntegration.stringrep(m8) == s8

    # Test 9: Scientific notation, err > val, N < 0
    s9 = "1.0432(1)e-10"
    m9 = measurement(s9)
    println("Test case: $s9")
    @test MCIntegration.stringrep(m9) == s9
end

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
    println(config)
    report(config)
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