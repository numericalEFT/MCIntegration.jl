# ---------------------------------------------------------------------------------------------------
# Static access to the vi-th variable of the (heterogeneous) variable tuple.  `config.var[vi]` with a
# runtime index has an abstract type, so every Dist call on it is a dynamic dispatch; the recursion
# below is unrolled by the compiler and each branch is concretely typed.
@inline _offset_at(vars::Tuple, vi::Int) = vi == 1 ? first(vars).offset : _offset_at(Base.tail(vars), vi - 1)
@inline _offset_at(::Tuple{}, vi::Int) = 0
@inline _single_discrete_at(vars::Tuple, vi::Int) = vi == 1 ? (first(vars) isa Discrete && first(vars).size == 1) : _single_discrete_at(Base.tail(vars), vi - 1)
@inline _single_discrete_at(::Tuple{}, vi::Int) = false
@inline _shift_at!(vars::Tuple, vi::Int, idx::Int, config) = vi == 1 ? Dist.shift!(first(vars), idx, config) : _shift_at!(Base.tail(vars), vi - 1, idx, config)
@inline _shift_at!(::Tuple{}, vi::Int, idx::Int, config) = 0.0
@inline _shiftRollback_at!(vars::Tuple, vi::Int, idx::Int, config) = vi == 1 ? (Dist.shiftRollback!(first(vars), idx, config); nothing) : _shiftRollback_at!(Base.tail(vars), vi - 1, idx, config)
@inline _shiftRollback_at!(::Tuple{}, vi::Int, idx::Int, config) = nothing
@inline _swap_at!(vars::Tuple, vi::Int, i1::Int, i2::Int, config) = vi == 1 ? Dist.swap!(first(vars), i1, i2, config) : _swap_at!(Base.tail(vars), vi - 1, i1, i2, config)
@inline _swap_at!(::Tuple{}, vi::Int, i1::Int, i2::Int, config) = 0.0
@inline _swapRollback_at!(vars::Tuple, vi::Int, i1::Int, i2::Int, config) = vi == 1 ? (Dist.swapRollback!(first(vars), i1, i2, config); nothing) : _swapRollback_at!(Base.tail(vars), vi - 1, i1, i2, config)
@inline _swapRollback_at!(::Tuple{}, vi::Int, i1::Int, i2::Int, config) = nothing
"""create (op = 1) / remove (2) / createRollback (3) / removeRollback (4) the slots `lo:hi` (plus offset) of the
vi-th variable; returns the product of the proposal factors (1 for the rollbacks)"""
@inline function _dof_op!(vars::Tuple, vi::Int, lo::Int, hi::Int, config, ::Val{op}) where {op}
    if vi == 1
        var = first(vars)
        offset = var.offset
        prop = 1.0
        for pos = lo:hi
            if op == 1
                prop *= Dist.create!(var, pos + offset, config)
            elseif op == 2
                prop *= Dist.remove!(var, pos + offset, config)
            elseif op == 3
                Dist.createRollback!(var, pos + offset, config)
            else
                Dist.removeRollback!(var, pos + offset, config)
            end
        end
        return prop
    else
        return _dof_op!(Base.tail(vars), vi - 1, lo, hi, config, Val(op))
    end
end
@inline _dof_op!(::Tuple{}, vi::Int, lo::Int, hi::Int, config, ::Val{op}) where {op} = 1.0
# ---------------------------------------------------------------------------------------------------

function changeIntegrand(config::Configuration{N,V,P,O,T}, integrand, state) where {N,V,P,O,T}
    # update to change an integrand to its neighbors. 
    # The degrees of freedom could be increase, decrease or remain the same.

    curr = state.curr
    new = rand(config.rng, config.neighbor[curr]) # jump to a randomly picked neighboring integrand
    (new == curr) && return

    currdof, newdof = config.dof[curr], config.dof[new]

    # propose probability caused by the selection of neighbors
    prop = length(config.neighbor[curr]) / length(config.neighbor[new])

    # create/remove variables if there are more/less degrees of freedom
    for vi = 1:length(config.var)
        if (currdof[vi] < newdof[vi]) # more degrees of freedom
            prop *= _dof_op!(config.var, vi, currdof[vi] + 1, newdof[vi], config, Val(1))
        elseif (currdof[vi] > newdof[vi]) # less degrees of freedom
            prop *= _dof_op!(config.var, vi, newdof[vi] + 1, currdof[vi], config, Val(2))
        end
    end

    # sampler may want to reject, then prop has already been set to zero
    if prop <= eps(0.0)
        return
    end

    # if new == config.norm, then newWeight will not be used, 
    # but still needs to be set to zero(T) so that newWeight is type stable
    newWeight =
        (new == config.norm) ?
        zero(T) :
        (fieldcount(V) == 1) ? integrand(new, config.var[1], config) : integrand(new, config.var, config)

    config.neval += 1
    # integrand_wrap(new, config, integrand)
    newProbability = (new == config.norm) ?
                     config.reweight[new] :
                     abs(newWeight) * config.reweight[new]

    R = prop * newProbability / state.probability

    config.propose[1, curr, new] += 1.0
    if rand(config.rng) < R  # accept the change
        config.accept[1, curr, new] += 1.0
        state.curr = new
        state.weight = newWeight
        state.probability = newProbability
    else # reject the change
        ############ Redo changes to config.var #############
        for vi = 1:length(config.var)
            if (currdof[vi] < newdof[vi]) # more degrees of freedom
                _dof_op!(config.var, vi, currdof[vi] + 1, newdof[vi], config, Val(3))
            elseif (currdof[vi] > newdof[vi]) # less degrees of freedom
                _dof_op!(config.var, vi, newdof[vi] + 1, currdof[vi], config, Val(4))
            end
        end
    end
    return
end

function changeVariable(config::Configuration{N,V,P,O,T}, integrand, state) where {N,V,P,O,T}
    # update to change the variables of the current diagrams
    (state.curr == config.norm) && return

    curr = state.curr
    currdof = config.dof[curr]
    vi = rand(config.rng, 1:length(currdof)) # update the variable type of the index vi
    if _single_discrete_at(config.var, vi) # there is only one discrete element, there is nothing to sample with.
        return
    end
    (currdof[vi] <= 0) && return # return if the var has zero degree of freedom
    idx = _offset_at(config.var, vi) + rand(config.rng, 1:currdof[vi]) # randomly choose one var to update

    prop = _shift_at!(config.var, vi, idx, config)

    # sampler may want to reject, then prop has already been set to zero
    if prop <= eps(0.0)
        return
    end

    weight = (fieldcount(V) == 1) ? integrand(curr, config.var[1], config) : integrand(curr, config.var, config)

    config.neval += 1

    newProbability = abs(weight) * config.reweight[curr]
    R = prop * newProbability / state.probability

    config.propose[2, curr, vi] += 1.0
    if rand(config.rng) < R
        config.accept[2, curr, vi] += 1.0
        state.weight = weight
        state.probability = newProbability
    else
        _shiftRollback_at!(config.var, vi, idx, config)
    end
    return
end

function swapVariable(config::Configuration{N,V,P,O,T}, integrand, state) where {N,V,P,O,T}
    # update to change the variables of the current diagrams
    (state.curr == config.norm) && return

    curr = state.curr
    currdof = config.dof[curr]
    vi = rand(config.rng, 1:length(currdof)) # update the variable type of the index vi
    (currdof[vi] <= 0) && return # return if the var has zero degree of freedom
    offset = _offset_at(config.var, vi)
    idx1 = offset + rand(config.rng, 1:currdof[vi]) # randomly choose one var to update
    idx2 = offset + rand(config.rng, 1:currdof[vi]) # randomly choose one var to update
    (idx1 == idx2) && return

    prop = _swap_at!(config.var, vi, idx1, idx2, config)

    # sampler may want to reject, then prop has already been set to zero
    if prop <= eps(0.0)
        return
    end

    weight = (fieldcount(V) == 1) ? integrand(curr, config.var[1], config) : integrand(curr, config.var, config)

    config.neval += 1

    newProbability = abs(weight) * config.reweight[curr]
    R = prop * newProbability / state.probability

    config.propose[3, curr, vi] += 1.0
    if rand(config.rng) < R
        config.accept[3, curr, vi] += 1.0
        state.weight = weight
        state.probability = newProbability
    else
        _swapRollback_at!(config.var, vi, idx1, idx2, config)
    end
    return
end