using StatsBase

@testset "Statistics" begin

    function series(block)
        @assert block > 1
        timeseries = rand(block)
        _sum = sum(timeseries)
        _sqaure_sum = sum(timeseries .^ 2)
        return timeseries, _sum, _sqaure_sum, StatsBase.mean(timeseries), std(timeseries)/sqrt(block)
        # standard deviation of the mean already contains a factor 1/sqrt(block-1)
    end

    block = 10
    series1, _sum1, _sqaure_sum1, mean1, error1 = series(10)
    series2, _sum2, _sqaure_sum2, mean2, error2 = series(10)

    #two scalar integrals
    obsSum = [_sum1, _sum2]
    obsSquaredSum = [_sqaure_sum1, _sqaure_sum2]

    mean, error = MCIntegration._mean_std(obsSum, obsSquaredSum, block)
    @test mean[1] ≈ mean1
    @test mean[2] ≈ mean2
    @test error[1] ≈ error1
    @test error[2] ≈ error2

    #one 2-vector integral
    mean, error = MCIntegration._mean_std([obsSum, ], [obsSquaredSum, ], block)
    @test mean[1][1] ≈ mean1
    @test mean[1][2] ≈ mean2
    @test error[1][1] ≈ error1
    @test error[1][2] ≈ error2

    #one complex integral in a vector
    _csum = _sum1 + im * _sum2
    _csqaure_sum = _sqaure_sum1 + im * _sqaure_sum2
    mean, error = MCIntegration._mean_std([_csum, ], [_csqaure_sum, ], block)
    @test mean[1] ≈ mean1 + mean2*im
    @test error[1] ≈ error1 + error2*im

    #on complex integrals
    _csum = _sum1 + im * _sum2
    _csqaure_sum = _sqaure_sum1 + im * _sqaure_sum2
    mean, error = MCIntegration._mean_std(_csum, _csqaure_sum, block)
    @test mean[1] ≈ mean1 + mean2*im
    @test error[1] ≈ error1 + error2*im

end