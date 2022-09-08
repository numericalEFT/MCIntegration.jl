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
        offset = config.var[vi].offset
        if (currdof[vi] < newdof[vi]) # more degrees of freedom
            for pos = currdof[vi]+1:newdof[vi]
                prop *= Dist.create!(config.var[vi], pos + offset, config)
            end
        elseif (currdof[vi] > newdof[vi]) # less degrees of freedom
            for pos = newdof[vi]+1:currdof[vi]
                prop *= Dist.remove!(config.var[vi], pos + offset, config)
            end
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
            offset = config.var[vi].offset
            if (currdof[vi] < newdof[vi]) # more degrees of freedom
                for pos = currdof[vi]+1:newdof[vi]
                    Dist.createRollback!(config.var[vi], pos + offset, config)
                end
            elseif (currdof[vi] > newdof[vi]) # less degrees of freedom
                for pos = newdof[vi]+1:currdof[vi]
                    Dist.removeRollback!(config.var[vi], pos + offset, config)
                end
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
    var = config.var[vi]
    if (var isa Discrete) && (var.size == 1) # there is only one discrete element, there is nothing to sample with.
        return
    end
    (currdof[vi] <= 0) && return # return if the var has zero degree of freedom
    idx = var.offset + rand(config.rng, 1:currdof[vi]) # randomly choose one var to update

    prop = Dist.shift!(var, idx, config)

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
        Dist.shiftRollback!(var, idx, config)
    end
    return
end

function swapVariable(config::Configuration{N,V,P,O,T}, integrand, state) where {N,V,P,O,T}
    # update to change the variables of the current diagrams
    (state.curr == config.norm) && return

    curr = state.curr
    currdof = config.dof[curr]
    vi = rand(config.rng, 1:length(currdof)) # update the variable type of the index vi
    var = config.var[vi]
    (currdof[vi] <= 0) && return # return if the var has zero degree of freedom
    idx1 = var.offset + rand(config.rng, 1:currdof[vi]) # randomly choose one var to update
    idx2 = var.offset + rand(config.rng, 1:currdof[vi]) # randomly choose one var to update
    (idx1 == idx2) && return

    prop = Dist.swap!(var, idx1, idx2, config)

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
        Dist.swapRollback!(var, idx1, idx2, config)
    end
    return
end