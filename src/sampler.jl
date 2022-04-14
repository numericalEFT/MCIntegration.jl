"""
    create!(newIdx::Int, size::Int, rng=GLOBAL_RNG)

Propose to generate new index (uniformly) randomly in [1, size]

# Arguments
- `newIdx`:  index ∈ [1, size]
- `size` : up limit of the index
- `rng=GLOBAL_RNG` : random number generator
"""
@inline function create!(d::Discrete, idx::Int, config)
    (idx >= length(d.data) - 1) && error("$idx overflow!")
    d[idx] = rand(config.rng, d.lower:d.upper)
    return Float64(d.upper - d.lower + 1) # lower:upper has upper-lower+1 elements!
end

@inline createRollback!(d::Discrete, idx::Int, config) = nothing

"""
    remove!(newIdx::Int, size::Int, rng=GLOBAL_RNG)

Propose to remove the old index in [1, size]

# Arguments
- `oldIdx`:  index ∈ [1, size]
- `size` : up limit of the index
- `rng=GLOBAL_RNG` : random number generator
"""
@inline function remove!(d::Discrete, idx::Int, config)
    (idx >= length(d.data) - 1) && error("$idx overflow!")
    return 1.0 / Float64(d.upper - d.lower + 1)
end

@inline removeRollback!(d::Discrete, idx::Int, config) = nothing

"""
    shift!(d::Discrete, idx::Int, config)

Propose to shift the old index in [1, size] to a new index

# Arguments
- `oldIdx`:  old index ∈ [1, size]
- `newIdx`:  new index ∈ [1, size], will be modified!
- `size` : up limit of the index
- `rng=GLOBAL_RNG` : random number generator
"""
@inline function shift!(d::Discrete, idx::Int, config)
    (idx >= length(d.data) - 1) && error("$idx overflow!")
    d[end] = d[idx] # save the current variable
    d[idx] = rand(config.rng, d.lower:d.upper)
    return 1.0
end

@inline function shiftRollback!(d::Discrete, idx::Int, config)
    (idx >= length(d.data) - 1) && error("$idx overflow!")
    d[idx] = d[end]
end

"""
    swap!(d::Discrete, idx1::Int, idx2::Int, config)

 Swap the variables idx1 and idx2

"""
@inline function swap!(d::Discrete, idx1::Int, idx2::Int, config)
    ((idx1 >= length(d.data) - 1) || (idx2 >= length(d.data) - 1)) && error("$idx overflow!")
    d[idx1], d[idx2] = d[idx2], d[idx1]
    return 1.0
end

@inline function swapRollback!(d::Discrete, idx1::Int, idx2::Int, config)
    ((idx1 >= length(d.data) - 1) || (idx2 >= length(d.data) - 1)) && error("$idx overflow!")
    d[idx1], d[idx2] = d[idx2], d[idx1]
end


"""
    create!(K::FermiK{D}, idx::Int, rng=GLOBAL_RNG)

Propose to generate new Fermi K in [Kf-δK, Kf+δK)

# Arguments
- `newK`:  vector of dimension of d=2 or 3
"""
function create!(K::FermiK{D}, idx::Int, config) where {D}
    @assert idx > K.offset
    (idx >= length(K.data) - 1) && error("$idx overflow!")
    rng = config.rng
    ############ Simple Way ########################
    # for i in 1:DIM
    #     newK[i] = Kf * (rand(rng) - 0.5) * 2.0
    # end
    # return (2.0 * Kf)^DIM
    ################################################

    Kamp = K.kF + (rand(rng) - 0.5) * 2.0 * K.δk
    (Kamp <= 0.0) && return 0.0
    # Kf-dK<Kamp<Kf+dK 
    ϕ = 2π * rand(rng)
    if D == 3 # dimension 3
        θ = π * rand(rng)
        # newK .= Kamp .* Mom(cos(ϕ) * sin(θ), sin(ϕ) * sin(θ), cos(θ))
        # K[idx] = @SVector [Kamp * cos(ϕ) * sin(θ), Kamp * sin(ϕ) * sin(θ), Kamp * cos(θ)]
        K.data[1, idx] = Kamp * cos(ϕ) * sin(θ)
        K.data[2, idx] = Kamp * sin(ϕ) * sin(θ)
        K.data[3, idx] = Kamp * cos(θ)
        return 2 * K.δk * 2π * π * (sin(θ) * Kamp^2)
        # prop density of KAmp in [Kf-dK, Kf+dK), prop density of Phi
        # prop density of Theta, Jacobian
    else  # DIM==2
        # K[idx] = @SVector [Kamp * cos(ϕ), Kamp * sin(ϕ)]
        K.data[1, idx] = Kamp * cos(ϕ)
        K.data[2, idx] = Kamp * sin(ϕ)
        return 2 * K.δk * 2π * Kamp
        # prop density of KAmp in [Kf-dK, Kf+dK), prop density of Phi, Jacobian
    end
end
createRollback!(K::FermiK{D}, idx::Int, config) where {D} = nothing

"""
    removeFermiK!(oldK, Kf=1.0, δK=0.5, rng=GLOBAL_RNG)

Propose to remove an existing Fermi K in [Kf-δK, Kf+δK)

# Arguments
- `oldK`:  vector of dimension of d=2 or 3
"""
function remove!(K::FermiK{D}, idx::Int, config) where {D}
    @assert idx > K.offset
    (idx >= length(K.data) - 1) && error("$idx overflow!")
    ############## Simple Way #########################
    # for i in 1:DIM
    #     if abs(oldK[i]) > Kf
    #         return 0.0
    #     end
    # end
    # return 1.0 / (2.0 * Kf)^DIM
    ####################################################

    oldK = K[idx]
    Kamp = sqrt(dot(oldK, oldK))
    if !(K.kF - K.δk < Kamp < K.kF + K.δk)
        return 0.0
    end
    # (Kamp < Kf - dK || Kamp > Kf + dK) && return 0.0
    if D == 3 # dimension 3
        sinθ = sqrt(oldK[1]^2 + oldK[2]^2) / Kamp
        sinθ < 1.0e-15 && return 0.0
        return 1.0 / (2 * K.δk * 2π * π * sinθ * Kamp^2)
    else  # DIM==2
        return 1.0 / (2 * K.δk * 2π * Kamp)
    end
end

removeRollback!(K::FermiK{D}, idx::Int, config) where {D} = nothing

"""
    shiftK!(oldK, newK, step, rng=GLOBAL_RNG)

Propose to shift oldK to newK. Work for generic momentum vector
"""
function shift!(K::FermiK{D}, idx::Int, config) where {D}
    @assert idx > K.offset
    (idx >= length(K.data) - 1) && error("$idx overflow!")
    K[end] = K[idx]  # save current K

    rng = config.rng
    x = rand(rng)
    if x < 1.0 / 3
        λ = 1.5
        ratio = 1.0 / λ + rand(rng) * (λ - 1.0 / λ)
        K[idx] = K[idx] * ratio
        return (D == 2) ? 1.0 : ratio
    elseif x < 2.0 / 3
        ϕ = rand(rng) * 2π
        if (D == 3)
            # sample uniformly on sphere, check http://corysimon.github.io/articles/uniformdistn-on-sphere/ 
            θ = acos(1 - 2 * rand(rng))
            Kamp = sqrt(K[idx][1]^2 + K[idx][2]^2 + K[idx][3]^2)
            # K[idx] = @SVector [Kamp * cos(ϕ) * sin(θ), Kamp * sin(ϕ) * sin(θ), Kamp * cos(θ)]
            K.data[1, idx] = Kamp * cos(ϕ) * sin(θ)
            K.data[2, idx] = Kamp * sin(ϕ) * sin(θ)
            K.data[3, idx] = Kamp * cos(θ)
            return 1.0
        else # D=2
            Kamp = sqrt(K[idx][1]^2 + K[idx][2]^2)
            # K = @SVector [Kamp * cos(ϕ), Kamp * sin(ϕ)]
            K.data[1, idx] = Kamp * cos(ϕ)
            K.data[2, idx] = Kamp * sin(ϕ)
            return 1.0
        end
    else
        Kc, dk = K[idx], K.δk
        if (D == 3)
            # K[idx] = @SVector [Kc[1] + (rand(rng) - 0.5) * dk, Kc[2] + (rand(rng) - 0.5) * dk, Kc[3] + (rand(rng) - 0.5) * dk]
            K.data[1, idx] = Kc[1] + (rand(rng) - 0.5) * dk
            K.data[2, idx] = Kc[2] + (rand(rng) - 0.5) * dk
            K.data[3, idx] = Kc[3] + (rand(rng) - 0.5) * dk
        else # D=2
            # K[idx] = @SVector [Kc[1] + (rand(rng) - 0.5) * dk, Kc[2] + (rand(rng) - 0.5) * dk]
            K.data[1, idx] = Kc[1] + (rand(rng) - 0.5) * dk
            K.data[2, idx] = Kc[2] + (rand(rng) - 0.5) * dk
        end
        # K[idx] += (rand(rng, D) .- 0.5) .* K.δk
        return 1.0
    end
end

function shiftRollback!(K::FermiK{D}, idx::Int, config) where {D}
    (idx >= length(K.data) - 1) && error("$idx overflow!")
    K[idx] = K[end]
end

@inline function swap!(K::FermiK{D}, idx1::Int, idx2::Int, config) where {D}
    ((idx1 >= length(K.data) - 1) || (idx2 >= length(K.data) - 1)) && error("$idx1 or $idx2 overflow!")
    if D == 2
        K.data[1, idx1], K.data[1, idx2] = K.data[1, idx2], K.data[1, idx1]
        K.data[2, idx1], K.data[2, idx2] = K.data[2, idx2], K.data[2, idx1]
    elseif D == 3
        K.data[1, idx1], K.data[1, idx2] = K.data[1, idx2], K.data[1, idx1]
        K.data[2, idx1], K.data[2, idx2] = K.data[2, idx2], K.data[2, idx1]
        K.data[3, idx1], K.data[3, idx2] = K.data[3, idx2], K.data[3, idx1]
    else
        error("not implemented!")
    end
    return 1.0
end

@inline function swapRollback!(K::FermiK{D}, idx1::Int, idx2::Int, config) where {D}
    ((idx1 >= length(K.data) - 1) || (idx2 >= length(K.data) - 1)) && error("$idx1 or $idx2 overflow!")
    if D == 2
        K.data[1, idx1], K.data[1, idx2] = K.data[1, idx2], K.data[1, idx1]
        K.data[2, idx1], K.data[2, idx2] = K.data[2, idx2], K.data[2, idx1]
    elseif D == 3
        K.data[1, idx1], K.data[1, idx2] = K.data[1, idx2], K.data[1, idx1]
        K.data[2, idx1], K.data[2, idx2] = K.data[2, idx2], K.data[2, idx1]
        K.data[3, idx1], K.data[3, idx2] = K.data[3, idx2], K.data[3, idx1]
    else
        error("not implemented!")
    end
end

"""
    create!(T::Tau, idx::Int, rng=GLOBAL_RNG)

Propose to generate new tau (uniformly) randomly in [0, β), return proposal probability

# Arguments
- `T`:  Tau variable
- `idx`: T.data[idx] will be updated
"""
@inline function create!(T::Tau, idx::Int, config)
    (idx >= length(T.data) - 1) && error("$idx overflow!")
    T[idx] = rand(config.rng) * T.β
    return T.β
end
@inline createRollback!(T::Tau, idx::Int, config) = nothing

"""
    remove(T::Tau, idx::Int, rng=GLOBAL_RNG)

Propose to remove old tau in [0, β), return proposal probability

# Arguments
- `T`:  Tau variable
- `idx`: T.data[idx] will be updated
"""
@inline function remove!(T::Tau, idx::Int, config)
    (idx >= length(T.data) - 1) && error("$idx overflow!")
    return 1.0 / T.β
end
@inline removeRollback!(T::Tau, idx::Int, config) = nothing

"""
    shift!(T::Tau, idx::Int, rng=GLOBAL_RNG)

Propose to shift an existing tau to a new tau, both in [0, β), return proposal probability

# Arguments
- `T`:  Tau variable
- `idx`: T.data[idx] will be updated
"""
@inline function shift!(T::Tau, idx::Int, config)
    (idx >= length(T.data) - 1) && error("$idx overflow!")
    T[end] = T[idx]
    rng = config.rng
    x = rand(rng)
    if x < 1.0 / 3
        T[idx] = T[idx] + 2 * T.λ * (rand(rng) - 0.5)
    elseif x < 2.0 / 3
        T[idx] = T.β - T[idx]
    else
        T[idx] = rand(rng) * T.β
    end

    if T[idx] < 0.0
        T[idx] += T.β
    elseif T[idx] > T.β
        T[idx] -= T.β
    end

    return 1.0
end

@inline function shiftRollback!(T::Tau, idx::Int, config)
    (idx >= length(T.data) - 1) && error("$idx overflow!")
    T[idx] = T[end]
end

"""
    swap!(T::Tau, idx1::Int, idx2::Int, config)

 Swap the variables idx1 and idx2

"""
@inline function swap!(T::Tau, idx1::Int, idx2::Int, config)
    ((idx1 >= length(T.data) - 1) || (idx2 >= length(T.data) - 1)) && error("$idx1 or $idx2 overflow!")
    T[idx1], T[idx2] = T[idx2], T[idx1]
    return 1.0
end

@inline function swapRollback!(T::Tau, idx1::Int, idx2::Int, config)
    ((idx1 >= length(T.data) - 1) || (idx2 >= length(T.data) - 1)) && error("$idx1 or $idx2 overflow!")
    T[idx1], T[idx2] = T[idx2], T[idx1]
end

"""
    create!(T::TauPair, idx::Int, rng=GLOBAL_RNG)

Propose to generate a new pair of tau (uniformly) randomly in [0, β), return proposal probability

# Arguments
- `T`:  TauPair variable
- `idx`: T.data[idx] will be updated
"""
@inline function create!(T::TauPair, idx::Int, config)
    (idx >= length(T.data) - 1) && error("$idx overflow!")
    rng = config.rng
    T[idx][1] = rand(rng) * T.β
    T[idx][2] = rand(rng) * T.β
    return T.β * T.β
end

@inline createRollback!(T::TauPair, idx::Int, config) = nothing


"""
    remove(T::TauPair, idx::Int, rng=GLOBAL_RNG)

Propose to remove an existing pair of tau in [0, β), return proposal probability

# Arguments
- `T`:  Tau variable
- `idx`: T.data[idx] will be updated
"""
@inline function remove!(T::TauPair, idx::Int, config)
    (idx >= length(T.data) - 1) && error("$idx overflow!")
    return 1.0 / T.β / T.β
end
@inline removeRollback!(T::TauPair, idx::Int, config) = nothing

"""
    shift!(T::TauPair, idx::Int, rng=GLOBAL_RNG)

Propose to shift an existing tau pair to a new tau pair, both in [0, β), return proposal probability

# Arguments
- `T`:  Tau variable
- `idx`: T.t[idx] will be updated
"""
@inline function shift!(T::TauPair, idx::Int, config)
    (idx >= length(T.data) - 1) && error("$idx overflow!")
    T[end] .= T[idx]
    rng = config.rng
    x = rand(rng)
    if x < 1.0 / 3
        T[idx][1] += 2 * T.λ * (rand(rng) - 0.5)
        T[idx][2] += 2 * T.λ * (rand(rng) - 0.5)
    elseif x < 2.0 / 3
        T[idx][1] = T.β - T[idx][1]
        T[idx][2] = T.β - T[idx][2]
    else
        T[idx][1] = rand(rng) * T.β
        T[idx][2] = rand(rng) * T.β
    end

    if T[idx][1] < 0.0
        T[idx][1] += T.β
    elseif T[idx][1] > T.β
        T[idx][1] -= T.β
    end

    if T[idx][2] < 0.0
        T[idx][2] += T.β
    elseif T[idx][2] > T.β
        T[idx][2] -= T.β
    end

    return 1.0
    # return 0.0
end

@inline function shiftRollback!(T::TauPair, idx::Int, config)
    (idx >= length(T.data) - 1) && error("$idx overflow!")
    T[idx] .= T[end]
end

@inline function swap!(T::TauPair, idx1::Int, idx2::Int, config)
    ((idx1 >= length(T.data) - 1) || (idx2 >= length(T.data) - 1)) && error("$idx1 or $idx2 overflow!")
    T[idx1], T[idx2] = T[idx2], T[idx1]
    return 1.0
end

@inline function swapRollback!(T::TauPair, idx1::Int, idx2::Int, config)
    ((idx1 >= length(T.data) - 1) || (idx2 >= length(T.data) - 1)) && error("$idx1 or $idx2 overflow!")
    T[idx1], T[idx2] = T[idx2], T[idx1]
end

"""
    create!(theta::Angle, idx::Int, rng=GLOBAL_RNG)

Propose to generate new angle (uniformly) randomly in [0, 2π), return proposal probability

# Arguments
- `theta`:  angle variable
- `idx`: theta.t[idx] will be updated
"""
@inline function create!(theta::Angle, idx::Int, config)
    (idx >= length(theta.data) - 1) && error("$idx overflow!")
    theta[idx] = rand(config.rng) * 2π
    return 2π
end
@inline createRollback!(theta::Angle, idx::Int, config) = nothing


"""
    remove(theta::Angle, idx::Int, rng=GLOBAL_RNG)

Propose to remove old theta in [0, 2π), return proposal probability

# Arguments
    - `theta`:  Tau variable
- `idx`: theta.t[idx] will be updated
"""
@inline function remove!(theta::Angle, idx::Int, config)
    (idx >= length(theta.data) - 1) && error("$idx overflow!")
    return 1.0 / 2.0 / π
end
@inline removeRollback!(theta::Angle, idx::Int, config) = nothing

"""
    shift!(theta::Angle, idx::Int, rng=GLOBAL_RNG)

Propose to shift the old theta to new theta, both in [0, 2π), return proposal probability

# Arguments
- `theta`:  angle variable
- `idx`: theta.t[idx] will be updated
"""
@inline function shift!(theta::Angle, idx::Int, config)
    (idx >= length(theta.data) - 1) && error("$idx overflow!")
    theta[end] = theta[idx]
    rng = config.rng
    x = rand(rng)
    if x < 1.0 / 3
        theta[idx] = theta[idx] + 2 * theta.λ * (rand(rng) - 0.5)
    elseif x < 2.0 / 3
        theta[idx] = 2π - theta[idx]
    else
        theta[idx] = rand(rng) * 2π
    end

    if theta[idx] < 0.0
        theta[idx] += 2π
    elseif theta[idx] > 2π
        theta[idx] -= 2π
    end

    return 1.0
end

@inline function shiftRollback!(theta::Angle, idx::Int, config)
    (idx >= length(theta.data) - 1) && error("$idx overflow!")
    theta[idx] = theta[end]
end



@inline function RFK_Weight(K::RadialFermiK, k)
    # norm = atan(K.kF/K.δk)/K.δk+π/2.0/K.δk
    # return 1.0/((k-K.kF)^2+K.δk^2)/norm
    if 0.0 <= k < K.kF - K.δk
        return 1.0 / 3.0 / (K.kF - K.δk)
    elseif k < K.kF + K.δk
        return 2.0 * K.δk / (3π) / ((k - K.kF)^2 + K.δk^2)
    else
        return 1.0 / 3 / (K.kF + K.δk) * exp(1 - k / (K.kF + K.δk))
    end
    return 0.0
end

@inline function RFK_Propose(K::RadialFermiK, x, y)
    # norm = atan(K.kF/K.δk)/K.δk+π/2.0/K.δk
    # return K.kF-K.δk*tan(atan(K.kF/K.δk)-K.δk*x*norm )
    if x < 1.0 / 3
        return y * (K.kF - K.δk)
    elseif x < 2.0 / 3
        return K.kF + K.δk * tan(π * (2.0 * y - 1) / 4.0)
    else
        return (K.kF + K.δk) * (1 - log1p(-y))
    end
end

"""
    create!(K::RadialFermiK, idx::Int, rng=GLOBAL_RNG)

Propose to generate new k randomly in [0, +inf), return proposal probability
k is generated uniformly on [0, K.kF-K.δk), Lorentzianly on [K.kF-K.δk,K.kF+K.δk),
and exponentially on [K.kF-K.δk, +inf).


# Arguments
- `K`:  k variable
- `idx`: K.data[idx] will be updated
"""
@inline function create!(K::RadialFermiK, idx::Int, config)
    (idx >= length(K.data) - 1) && error("$idx overflow!")
    rng = config.rng
    x = rand(rng)
    y = rand(rng)
    K[idx] = RFK_Propose(K, x, y)
    if isnan(K[idx])
        return 0.0
    end
    return 1.0 / RFK_Weight(K, K[idx])
end
@inline createRollback!(K::RadialFermiK, idx::Int, config) = nothing

"""
    remove(K::RadialFermiK, idx::Int, rng=GLOBAL_RNG)

Propose to remove old k in [0, +inf), return proposal probability

# Arguments
- `K`:  K variable
- `idx`: K.data[idx] will be updated
"""
@inline function remove!(K::RadialFermiK, idx::Int, config)
    (idx >= length(K.data) - 1) && error("$idx overflow!")
    return RFK_Weight(K, K[idx])
end
@inline removeRollback!(K::RadialFermiK, idx::Int, config) = nothing

"""
    shift!(K::RadialFermiK, idx::Int, rng=GLOBAL_RNG)

Propose to shift an existing k to a new k, both in [0, +inf), return proposal probability

# Arguments
- `K`:  K variable
- `idx`: K.data[idx] will be updated
"""
@inline function shift!(K::RadialFermiK, idx::Int, config)
    (idx >= length(K.data) - 1) && error("$idx overflow!")
    K[end] = K[idx]
    rng = config.rng
    # K[idx] = RFK_Propose(K, rand(rng), rand(rng))
    # return RFK_Weight(K, K[end])/RFK_Weight(K, K[idx])
    x = rand(rng)
    if x < 1.0 / 3
        K[idx] = K[idx] + 2 * K.δk * (rand(rng) - 0.5)
        if K[idx] < 0.0
            K[idx] = -K[idx]
        end
        return 1.0
    elseif x < 2.0 / 3
        if K[idx] < K.kF - K.δk
            y = (K.kF - K.δk - K[idx]) / (K.kF - K.δk)
            K[idx] = RFK_Propose(K, 1.0, y)
        elseif K[idx] < K.kF + K.δk
            K[idx] = 2 * K.kF - K[idx]
        else
            y = exp(1 - K[idx] / (K.kF + K.δk))
            K[idx] = RFK_Propose(K, 0.0, y)
        end
        if isnan(K[idx])
            return 0.0
        end
        return RFK_Weight(K, K[end]) / RFK_Weight(K, K[idx])
    else
        K[idx] = RFK_Propose(K, rand(rng), rand(rng))
        if isnan(K[idx])
            return 0.0
        end
        return RFK_Weight(K, K[end]) / RFK_Weight(K, K[idx])
    end

end

@inline function shiftRollback!(K::RadialFermiK, idx::Int, config)
    (idx >= length(K.data) - 1) && error("$idx overflow!")
    K[idx] = K[end]
end

@inline function swap!(K::RadialFermiK, idx1::Int, idx2::Int, config)
    ((idx1 >= length(K.data) - 1) || (idx2 >= length(K.data) - 1)) && error("$idx1 or $idx2 overflow!")
    K[idx1], K[idx2] = K[idx2], K[idx1]
    return 1.0
end

@inline function swapRollback!(K::RadialFermiK, idx1::Int, idx2::Int, config)
    ((idx1 >= length(K.data) - 1) || (idx2 >= length(K.data) - 1)) && error("$idx1 or $idx2 overflow!")
    K[idx1], K[idx2] = K[idx2], K[idx1]
end

"""
    create!(T::Continuous, idx::Int, rng=GLOBAL_RNG)

Propose to generate new (uniform) variable randomly in [T.lower, T.lower+T.range), return proposal probability

# Arguments
- `T`:  Continuous variable
- `idx`: T.data[idx] will be updated
"""
@inline function create!(T::Continuous, idx::Int, config)
    (idx >= length(T.data) - 1) && error("$idx overflow!")
    T[idx] = rand(config.rng) * T.range + T.lower
    return T.range
end
@inline createRollback!(T::Continuous, idx::Int, config) = nothing

"""
    remove(T::Continuous, idx::Int, rng=GLOBAL_RNG)

Propose to remove old variable in [T.lower, T.lower+T.range), return proposal probability

# Arguments
- `T`:  Continuous variable
- `idx`: T.data[idx] will be updated
"""
@inline function remove!(T::Continuous, idx::Int, config)
    (idx >= length(T.data) - 1) && error("$idx overflow!")
    return 1.0 / T.range
end
@inline removeRollback!(T::Continuous, idx::Int, config) = nothing

"""
    shift!(T::Continuous, idx::Int, rng=GLOBAL_RNG)

Propose to shift an existing variable to a new one, both in [T.lower, T.lower+T.range), return proposal probability

# Arguments
- `T`:  Continuous variable
- `idx`: T.data[idx] will be updated
"""
@inline function shift!(T::Continuous, idx::Int, config)
    (idx >= length(T.data) - 1) && error("$idx overflow!")
    T[end] = T[idx]
    rng = config.rng
    x = rand(rng)
    if x < 1.0 / 2
        T[idx] = T[idx] + 2 * T.λ * (rand(rng) - 0.5)
    else
        T[idx] = rand(rng) * T.range + T.lower
    end

    if T[idx] < T.lower
        T[idx] += T.range
    elseif T[idx] > T.lower + T.range
        T[idx] -= T.range
    end

    return 1.0
end

@inline function shiftRollback!(T::Continuous, idx::Int, config)
    (idx >= length(T.data) - 1) && error("$idx overflow!")
    T[idx] = T[end]
end


@inline function swap!(T::Continuous, idx1::Int, idx2::Int, config)
    ((idx1 >= length(T.data) - 1) || (idx2 >= length(T.data) - 1)) && error("$idx1 or $idx2 overflow!")
    T[idx1], T[idx2] = T[idx2], T[idx1]
    return 1.0
end

@inline function swapRollback!(T::Continuous, idx1::Int, idx2::Int, config)
    ((idx1 >= length(T.data) - 1) || (idx2 >= length(T.data) - 1)) && error("$idx1 or $idx2 overflow!")
    T[idx1], T[idx2] = T[idx2], T[idx1]
end