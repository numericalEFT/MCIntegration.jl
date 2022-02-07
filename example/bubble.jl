# This example demonstrated how to calculate the bubble diagram of free electrons using the Monte Carlo module

using LinearAlgebra, Random, Printf, BenchmarkTools, InteractiveUtils, Parameters
using ElectronGas
using StaticArrays
using Lehmann
using MCIntegration
# using ProfileView

const Steps = 1e6

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
    Qsize::Int = 16
    extQ::Vector{SVector{3,Float64}} = [@SVector [q, 0.0, 0.0] for q in LinRange(0.0, 3.0 * kF, Qsize)]
end

function integrand(config)
    if config.curr != 1
        error("impossible")
    end
    para = config.para

    T, K, Ext = config.var[1], config.var[2], config.var[3]
    k = K[1]
    Tin, Tout = T[1], T[2]
    extidx = Ext[1]
    q = para.extQ[extidx] # external momentum
    kq = k + q
    τ = (Tout - Tin)
    ω1 = (dot(k, k) - kF^2) / (2me)
    g1 = Spectral.kernelFermiT(τ, ω1, β)
    ω2 = (dot(kq, kq) - kF^2) / (2me)
    g2 = Spectral.kernelFermiT(-τ, ω2, β)
    phase = 1.0 / (2π)^3
    return g1 * g2 * spin * phase * cos(2π * para.n * τ / β) / β
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

    T = MCIntegration.Tau(β, β / 2.0)
    K = MCIntegration.FermiK(3, kF, 0.2 * kF, 10.0 * kF)
    Ext = MCIntegration.Discrete(1, length(extQ)) # external variable is specified

    dof = [[2, 1, 1],] # degrees of freedom of the normalization diagram and the bubble
    obs = zeros(Float64, Qsize) # observable for the normalization diagram and the bubble

    config = MCIntegration.Configuration(steps, (T, K, Ext), dof, obs; para = para)
    avg, std = MCIntegration.sample(config, integrand, measure; print = 0, Nblock = 16)
    # @profview MonteCarlo.sample(config, integrand, measure; print=0, Nblock=1)
    # sleep(100)

    if isnothing(avg) == false
        @unpack n, extQ = Para()

        for (idx, q) in enumerate(extQ)
            q = q[1]
            p = Polarization.Polarization0_ZeroTemp(q, para.n, basic) * spin
            @printf("%10.6f  %10.6f ± %10.6f  %10.6f\n", q / basic.kF, avg[idx], std[idx], p)
        end
    end
end

run(Steps)
# @time run(Steps)