@testset "Threads" begin
    if Threads.nthreads() == 1
        @warn "There is only one thread currently. You may want run julia wiht -t auto to enable multithreading."
    end

    Threads.@threads for i in 1:3
        println("test thread: ", i)
        X = Continuous(0.0, 1.0)
        config = Configuration((X,))
        result = sample(config, config -> (config.var[1][1])^i)
        println(i, " -> ", result.mean, "+-", result.stdev)
        check(result.mean, result.stdev, 1 / (1 + i))
    end
end