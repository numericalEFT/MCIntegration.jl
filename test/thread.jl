@testset "outer Threads" begin
    if Threads.nthreads() == 1
        @warn "There is only one thread currently. You may want run julia with -t auto to enable multithreading."
    end

    mean = zeros(3, 3)
    error = zeros(3, 3)
    Threads.@threads for i in 1:3
        id = Threads.threadid()
        println("test thread: $id for the index $i")
        X = Continuous(0.0, 1.0)

        result = integrate((X, c) -> (X[1])^i; var=(Continuous(0.0, 1.0),), dof=[[1,],], print=-1, solver=:vegas, parallel = :nothread)
        println(i, " -> ", result.mean, "+-", result.stdev)
        mean[i, 1] = result.mean
        error[i, 1] = result.stdev

        result = integrate((X, c) -> (X[1])^i; var=(Continuous(0.0, 1.0),), dof=[[1,],], print=-1, solver=:vegasmc, parallel = :nothread)
        println(i, " -> ", result.mean, "+-", result.stdev)
        mean[i, 2] = result.mean
        error[i, 2] = result.stdev

        result = integrate((idx, X, c) -> (X[1])^i; var=(Continuous(0.0, 1.0),), dof=[[1,],], print=-1, solver=:mcmc, parallel = :nothread)
        println(i, " -> ", result.mean, "+-", result.stdev)
        mean[i, 3] = result.mean
        error[i, 3] = result.stdev
    end
    
    for i in 1:3
        println("test outer threads for vegas")
        check(mean[i, 1], error[i, 1], 1 / (1 + i))
        println("test outer threads for vegasmc")
        check(mean[i, 2], error[i, 2], 1 / (1 + i))
        println("test outer threads for mcmc")
        check(mean[i, 3], error[i, 3], 1 / (1 + i))
    end

end

@testset "inner thread test" begin
    if Threads.nthreads() == 1
        @warn "There is only one thread currently. You may want run julia with -t auto to enable multithreading."
    end
    X = Continuous(0.0, 1.0)
    i = 2
    result = integrate((x, c) -> (x[1])^i; var=(Continuous(0.0, 1.0),), dof=[[1,],], print=-1, solver=:vegas, parallel = :thread)
    println(i, " -> ", result.mean, "+-", result.stdev)
    check(result, 1 / (1 + i))

    result = integrate((x, c) -> (x[1])^i; var=(Continuous(0.0, 1.0),), dof=[[1,],], print=-1, solver=:vegasmc, parallel = :thread)
    println(i, " -> ", result.mean, "+-", result.stdev)
    check(result, 1 / (1 + i))

    result = integrate((id, x, c) -> (x[1])^i; var=(Continuous(0.0, 1.0),), dof=[[1,],], print=-1, solver=:mcmc, parallel = :thread)
    println(i, " -> ", result.mean, "+-", result.stdev)
    check(result, 1 / (1 + i))
end