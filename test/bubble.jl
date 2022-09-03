# This example demonstrated how to calculate the bubble diagram of free electrons using the Monte Carlo module

using LinearAlgebra, Random, Printf, BenchmarkTools, InteractiveUtils, Parameters
using StaticArrays
using Lehmann
using MCIntegration
# using ProfileView

const Steps = 1e6

# include("parameter.jl")
@with_kw struct Para
    rs::Float64 = 1.0
    beta::Float64 = 25.0
    spin::Int = 2
    Qsize::Int = 8
    # n::Int = 0 # external Matsubara frequency
    dim::Int = 3
    me::Float64 = 0.5

    kF::Float64 = (dim == 3) ? (9π / (2spin))^(1 / 3) / rs : sqrt(4 / spin) / rs
    extQ::Vector{SVector{3,Float64}} = [@SVector [q, 0.0, 0.0] for q in LinRange(0.0 * kF, 2.0 * kF, Qsize)]
    β::Float64 = beta / (kF^2 / 2me)
end

function lindhard(q, para)
    me, kF, β, spin = para.me, para.kF, para.β, para.spin
    density = me * kF / (2π^2)
    # check sign of q, use -q if negative
    (q < 1e-6) && (q = 1e-6)
    x = q / 2 / kF
    if abs(q - 2 * kF) > EPS
        Π = density * (1 + (1 - x^2) * log1p(4 * x / ((1 - x)^2)) / 4 / x)
    else
        Π = density
    end
    return -Π * spin / 2
end

function integrand(T, K, Ext, config)
    # @assert idx == 1 "$(idx) is not a valid integrand"
    para, _Ext = config.userdata
    kF, β, me = para.kF, para.β, para.me
    k = K[1]
    # Tin, Tout = T[1], T[2]
    Tin, Tout = 0.0, T[1]
    extidx = Ext[1]
    q = para.extQ[extidx] # external momentum
    kq = k + q
    τ = (Tout - Tin)
    ω1 = (dot(k, k) - kF^2) / (2me)
    g1 = Spectral.kernelFermiT(τ, ω1, β)
    ω2 = (dot(kq, kq) - kF^2) / (2me)
    g2 = Spectral.kernelFermiT(-τ, ω2, β)
    phase = 1.0 / (2π)^3
    return g1 * g2 * para.spin * phase * cos(2π * para.n * τ / β)
end

function measure(obs, weight, config)
    # @assert idx == 1 "$(idx) is not a valid integrand"
    para, Ext = config.userdata
    obs[Ext[1]] += weight[1]
end

function run(steps)

    para = Para()
    @unpack extQ, Qsize = para
    kF, β = para.kF, para.β

    T = MCIntegration.Continuous(0.0, β; alpha=3.0, adapt=true)
    K = MCIntegration.FermiK(3, kF, 0.2 * kF, 10.0 * kF)
    Ext = MCIntegration.Discrete(1, length(extQ); adapt=true) # external variable is specified

    dof = [[1, 1, 1],] # degrees of freedom of the normalization diagram and the bubble
    obs = zeros(Float64, Qsize) # observable for the normalization diagram and the bubble

    # config = MCIntegration.Configuration(var=(T, K, Ext), dof=dof, obs=obs, para=para)
    result = MCIntegration.integrate(integrand; measure=measure, userdata=(para, Ext),
        var=(T, K, Ext), dof=dof, obs=obs, solver=:vegas,
        neval=steps, print=0, block=16)

    if isnothing(result) == false
        @unpack n, extQ = Para()
        avg, std = result.mean, result.stdev

        @printf("%10s  %10s   %10s  %10s\n", "q/kF", "avg", "err", "exact")
        for (idx, q) in enumerate(extQ)
            q = q[1]
            p = lindhard(q, para)
            @printf("%10.6f  %10.6f ± %10.6f  %10.6f\n", q / basic.kF, avg[idx], std[idx], p)
        end
        # println(MCIntegration.summary(result))
        # i = 1
        # println(result.config.var[i].histogram)
        # println(sum(result.config.var[i].histogram))
        # println(result.config.var[i].accumulation)
        # println(result.config.var[i].distribution)
    end
end

run(Steps)
# @time run(Steps)