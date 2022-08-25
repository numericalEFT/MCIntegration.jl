# This example demonstrated how to calculate the bubble diagram of free electrons using the Monte Carlo module

using LinearAlgebra, Random, Printf, BenchmarkTools, InteractiveUtils, Parameters
using ElectronGas
using StaticArrays
using Lehmann
using MCIntegration
# using ProfileView

const Steps = 1e7

# include("parameter.jl")
beta = 25.0
rs = 1.0
const basic = Parameter.rydbergUnit(1 / beta, rs, 3)
const β = basic.β
const kF = basic.kF
const me = basic.me
const spin = basic.spin

@with_kw struct Para
    n::Int = 0 # external Matsubara frequency
    Qsize::Int = 8
    extQ::Vector{SVector{3,Float64}} = [@SVector [q, 0.0, 0.0] for q in LinRange(0.0 * kF, 2.0 * kF, Qsize)]
end

function integrand(config)
    if config.curr != 1
        error("impossible")
    end
    para = config.para

    T, K, Ext = config.var[1], config.var[2], config.var[3]
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
    return g1 * g2 * spin * phase * cos(2π * para.n * τ / β)
end

function measure(config)
    obs = config.observable
    factor = 1.0 / config.reweight[config.curr]
    extidx = config.var[3][1]
    weight = integrand(config)
    obs[extidx] += weight / abs(weight) * factor
end

function run(steps)

    para = Para()
    @unpack extQ, Qsize = para

    # T = MCIntegration.Tau(β, β / 2.0)
    T = MCIntegration.Continuous(0.0, β; alpha=3.0, adapt=true)
    K = MCIntegration.FermiK(3, kF, 0.2 * kF, 10.0 * kF)
    Ext = MCIntegration.Discrete(1, length(extQ); adapt=true) # external variable is specified

    dof = [[1, 1, 1],] # degrees of freedom of the normalization diagram and the bubble
    obs = zeros(Float64, Qsize) # observable for the normalization diagram and the bubble

    config = MCIntegration.Configuration(var=(T, K, Ext), dof=dof, obs=obs, para=para)
    result = MCIntegration.sample(config, integrand, measure; neval=steps, print=0, block=16)

    if isnothing(result) == false
        @unpack n, extQ = Para()
        avg, std = result.mean, result.stdev

        @printf("%10s  %10s   %10s  %10s\n", "q/kF", "avg", "err", "exact")
        for (idx, q) in enumerate(extQ)
            q = q[1]
            p = Polarization.Polarization0_ZeroTemp(q, para.n, basic) * spin
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