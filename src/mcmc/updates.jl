function changeIntegrand(config, integrand)
    # update to change an integrand to its neighbors. 
    # The degrees of freedom could be increase, decrease or remain the same.

    curr = config.curr
    new = rand(config.rng, config.neighbor[curr]) # jump to a randomly picked neighboring integrand

    currProbability = config.probability

    # propose probability caused by the selection of neighbors
    prop = length(config.neighbor[curr]) / length(config.neighbor[new])

    # sampler may want to reject, then prop has already been set to zero
    if prop <= eps(0.0)
        return
    end

    config.curr = new
    weights = integrand(config)
    newProbability = (new == config.norm) ? config.reweight[new] : abs(weights[new]) / Dist.probability(config, new) * config.reweight[new]
    # R = prop * newAbsWeight * config.reweight[new] / currAbsWeight / config.reweight[curr]
    R = prop * newProbability / currProbability

    config.propose[1, curr, new] += 1.0
    if rand(config.rng) < R  # accept the change
        config.accept[1, curr, new] += 1.0
        setWeight!(config, weights)
    else # reject the change
        config.curr = curr # reset the current diagram index
    end
    return
end

function changeVariable(config, integrand)
    # update to change the variables of the current diagrams
    curr = config.curr
    # currdof = config.dof[curr]
    currdof = config.dof[1]
    vi = rand(config.rng, 1:length(currdof)) # update the variable type of the index vi
    var = config.var[vi]
    (currdof[vi] <= 0) && return # return if the var has zero degree of freedom
    idx = var.offset + rand(config.rng, 1:currdof[vi]) # randomly choose one var to update

    # oldvar = copy(var[idx])
    currProbability = config.probability

    prop = Dist.shift!(var, idx, config)

    # sampler may want to reject, then prop has already been set to zero
    if prop <= eps(0.0)
        return
    end

    weights = integrand(config)
    # newProbability = (curr == config.norm) ? config.reweight[curr] : abs(weights[curr]) / Dist.probability(config, curr) * config.reweight[curr]
    newProbability = (curr == config.norm) ? config.reweight[curr] : abs(weights[curr]) * config.reweight[curr]

    R = prop * newProbability / currProbability
    # R = newProbability / currProbability
    # println(prop, ", ", newProbability, ", ", currProbability, ", ", R)

    # curr == 2 && println("propose, $curr: old: $oldvar --> new: $(var[idx]), with R $newAbsWeight / $currAbsWeight * $prop = $R")
    config.propose[2, curr, vi] += 1.0
    if rand(config.rng) < R
        # curr == 2 && println("accept, $curr")
        config.accept[2, curr, vi] += 1.0
        setWeight!(config, weights)
        config.probability = newProbability
    else
        Dist.shiftRollback!(var, idx, config)
    end
    return
end

function swapVariable(config, integrand)
    # update to change the variables of the current diagrams
    curr = config.curr
    currdof = config.dof[1]
    vi = rand(config.rng, 1:length(currdof)) # update the variable type of the index vi
    var = config.var[vi]
    (currdof[vi] <= 0) && return # return if the var has zero degree of freedom
    idx1 = var.offset + rand(config.rng, 1:currdof[vi]) # randomly choose one var to update
    idx2 = var.offset + rand(config.rng, 1:currdof[vi]) # randomly choose one var to update
    (idx1 == idx2) && return

    currProbability = config.probability

    prop = Dist.swap!(var, idx1, idx2, config)

    # sampler may want to reject, then prop has already been set to zero
    if prop <= eps(0.0)
        return
    end

    weights = integrand(config)
    newProbability = (curr == config.norm) ? config.reweight[curr] : abs(weights[curr]) / Dist.probability(config, curr) * config.reweight[curr]
    R = prop * newProbability / currProbability

    config.propose[2, curr, vi] += 1.0
    if rand(config.rng) < R
        config.accept[2, curr, vi] += 1.0
        setWeight!(config, weights)
    else
        Dist.swapRollback!(var, idx1, idx2, config)
    end
    return
end