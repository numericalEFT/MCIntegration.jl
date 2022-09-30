# This example demonstrated how to calculate the bubble diagram of free electrons using the Monte Carlo module

using LinearAlgebra, Random, Printf
using StaticArrays
# using MCIntegration
# using ProfileView

@testset "Free electron polarization" begin
    Steps = 1e5

    # include("parameter.jl")
    Base.@kwdef struct Para
        rs::Float64 = 1.0
        beta::Float64 = 25.0
        spin::Int = 2
        Qsize::Int = 4
        # n::Int = 0 # external Matsubara frequency
        dim::Int = 3
        me::Float64 = 0.5

        kF::Float64 = (dim == 3) ? (9π / (2spin))^(1 / 3) / rs : sqrt(4 / spin) / rs
        extQ::Vector{SVector{3,Float64}} = [@SVector [q, 0.0, 0.0] for q in LinRange(0.0 * kF, 1.5 * kF, Qsize)]
        β::Float64 = beta / (kF^2 / 2me)
    end

    function lindhard(q, para) #free electron polarization
        me, kF, β, spin = para.me, para.kF, para.β, para.spin
        density = me * kF / (2π^2)
        # check sign of q, use -q if negative
        (q < 1e-6) && (q = 1e-6)
        x = q / 2 / kF
        if abs(q - 2 * kF) > 1e-6
            Π = (1 + (1 - x^2) * log1p(4 * x / ((1 - x)^2)) / 4 / x)
        else
            Π = 1.0
        end
        return -Π * density * spin / 2
    end

    function green(τ::T, ω::T, β::T) where {T}
        if τ >= T(0.0)
            return ω > T(0.0) ?
                   exp(-ω * τ) / (1 + exp(-ω * β)) :
                   exp(ω * (β - τ)) / (1 + exp(ω * β))
        else
            return ω > T(0.0) ?
                   -exp(-ω * (τ + β)) / (1 + exp(-ω * β)) :
                   -exp(-ω * τ) / (1 + exp(ω * β))
        end
    end

    function integrand(vars, config)
        # @assert idx == 1 "$(idx) is not a valid integrand"
        R, Theta, Phi, T, Ext = vars
        para = config.userdata
        kF, β, me = para.kF, para.β, para.me

        r = R[1] / (1 - R[1])
        θ = Theta[1]
        ϕ = Phi[1]
        # varK[:, i+1] .= [r * sin(θ) * cos(ϕ), r * sin(θ) * sin(ϕ), r * cos(θ)]
        k = [r * sin(θ) * cos(ϕ), r * sin(θ) * sin(ϕ), r * cos(θ)]
        factor = 1.0 / (2π)^(para.dim)  #each momentum loop is ∫dkxdkydkz/(2π)^3
        factor *= r^2 / (1 - R[1])^2 * sin(θ)

        Tin, Tout = 0.0, T[1]
        extidx = Ext[1]
        q = para.extQ[extidx] # external momentum
        kq = k + q
        τ = (Tout - Tin)
        ω1 = (dot(k, k) - kF^2) / (2me)
        g1 = green(τ, ω1, β)
        ω2 = (dot(kq, kq) - kF^2) / (2me)
        g2 = green(-τ, ω2, β)
        n = 0 # external Matsubara frequency
        return g1 * g2 * para.spin * factor * cos(2π * n * τ / β)
    end

    function integrand(idx, vars, config)
        return integrand(vars, config)::Float64
    end

    function measure(vars, obs, weight, config)
        # para = config.userdata
        Ext = vars[end]
        obs[1][Ext[1]] += weight[1]
    end
    function measure(idx, vars, obs, weight, config)
        # @assert idx == 1 "$(idx) is not a valid integrand"
        measure(vars, obs, weight, config)
    end

    function run(steps, alg, ratio)
        para = Para()
        extQ, Qsize = para.extQ, para.Qsize
        kF, β = para.kF, para.β

        T = Continuous(0.0, β; alpha=3.0, adapt=true)
        R = Continuous(0.0, 1.0; alpha=3.0, adapt=true)
        θ = Continuous(0.0, 1π; alpha=3.0, adapt=true)
        ϕ = Continuous(0.0, 2π; alpha=3.0, adapt=true)
        # K = MCIntegration.FermiK(3, kF, 0.2 * kF, 10.0 * kF)
        Ext = Discrete(1, length(extQ); adapt=false) # external variable is specified

        dof = [[1, 1, 1, 1, 1],] # degrees of freedom of the normalization diagram and the bubble
        obs = [zeros(Float64, Qsize),] # observable for the normalization diagram and the bubble

        result = integrate(integrand; measure=measure, userdata=para,
            var=(R, θ, ϕ, T, Ext), dof=dof, obs=obs, solver=alg,
            neval=steps, print=-1, block=8)
        @time result = integrate(integrand; measure=measure, userdata=para,
            var=(R, θ, ϕ, T, Ext), dof=dof, obs=obs, solver=alg,
            neval=steps * 10, print=0, block=64, niter=1, config=result.config, debug=true)

        if isnothing(result) == false
            avg, std = result.mean, result.stdev

            println("Algorithm : $(alg)")
            @printf("%10s  %10s   %10s  %10s\n", "q/kF", "avg", "err", "exact")
            for (idx, q) in enumerate(extQ)
                q = q[1]
                p = lindhard(q, para)
                @printf("%10.6f  %10.6f ± %10.6f  %10.6f\n", q / kF, avg[idx], std[idx], p)
                check(avg[idx], std[idx], p, ratio)
            end
            # check(avg[1], std[1], lindhard(extQ[1][1], para))
        end
    end

    run(Steps, :mcmc, 10.0)
    run(Steps, :vegas, 20.0)
    run(Steps, :vegasmc, 10.0)
    # run(Steps, :vegasmc) #currently vegasmc can not handle this 
    # @time run(Steps)
end