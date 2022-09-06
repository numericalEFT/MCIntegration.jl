function changeIntegrand(config::Configuration{N,V,P,O,T}, integrand) where {N,V,P,O,T}
    # update to change an integrand to its neighbors. 
    # The degrees of freedom could be increase, decrease or remain the same.

    curr = config.curr
    new = rand(config.rng, config.neighbor[curr]) # jump to a randomly picked neighboring integrand
    (new == curr) && return

    currdof, newdof = config.dof[curr], config.dof[new]

    currProbability = config.probability

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

    # config will be passed to user-defined integrand
    # the first parameter of the integrand function will be set idx = new
    # in case the user wants to explictly use config.curr instead of the idx parameter
    # let's make config.curr the same as idx, a.k.a. new
    config.curr = new

    # if new == config.norm, then newWeight will not be used, 
    # but still needs to be set to zero(T) so that newWeight is type stable
    newWeight =
        (new == config.norm) ?
        zero(T) :
        (fieldcount(V) == 1) ? integrand(new, config.var[1], config) : integrand(new, config.var, config)
    # integrand_wrap(new, config, integrand)
    newProbability = (new == config.norm) ?
                     config.reweight[new] :
                     abs(newWeight) * config.reweight[new]

    R = prop * newProbability / currProbability

    config.propose[1, curr, new] += 1.0
    if rand(config.rng) < R  # accept the change
        config.accept[1, curr, new] += 1.0
        if new != config.norm
            config.weights[new] = newWeight
        end
        config.probability = newProbability
        # config.absWeight = newAbsWeight
        # setweight!(config, weight)
    else # reject the change
        config.curr = curr # reset the current diagram index

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

function changeVariable(config::Configuration{N,V,P,O,T}, integrand) where {N,V,P,O,T}
    # update to change the variables of the current diagrams
    (config.curr == config.norm) && return

    curr = config.curr
    currdof = config.dof[curr]
    vi = rand(config.rng, 1:length(currdof)) # update the variable type of the index vi
    var = config.var[vi]
    (currdof[vi] <= 0) && return # return if the var has zero degree of freedom
    idx = var.offset + rand(config.rng, 1:currdof[vi]) # randomly choose one var to update

    # oldvar = copy(var[idx])
    # currAbsWeight = config.absWeight
    currProbability = config.probability

    prop = Dist.shift!(var, idx, config)

    # sampler may want to reject, then prop has already been set to zero
    if prop <= eps(0.0)
        return
    end

    # weight = integrand_wrap(curr, config, integrand)
    weight = (fieldcount(V) == 1) ? integrand(curr, config.var[1], config) : integrand(curr, config.var, config)
    newProbability = abs(weight) * config.reweight[curr]
    R = prop * newProbability / currProbability
    # newAbsWeight = abs(weight)
    # R = prop * newAbsWeight / currAbsWeight

    config.propose[2, curr, vi] += 1.0
    if rand(config.rng) < R
        # curr == 2 && println("accept, $curr")
        config.accept[2, curr, vi] += 1.0
        config.weights[curr] = weight
        config.probability = newProbability
        # config.absWeight = newAbsWeight
        # config.relativeWeight = weight / newAbsWeight / config.reweight[config.curr]
        # setweight!(config, weight)
        # accumulate!(var, idx)
    else
        # var[idx] = oldvar
        # config.absWeight = currAbsWeight
        Dist.shiftRollback!(var, idx, config)
    end
    return
end

function swapVariable(config::Configuration{N,V,P,O,T}, integrand) where {N,V,P,O,T}
    # update to change the variables of the current diagrams
    (config.curr == config.norm) && return

    curr = config.curr
    currdof = config.dof[curr]
    vi = rand(config.rng, 1:length(currdof)) # update the variable type of the index vi
    var = config.var[vi]
    (currdof[vi] <= 0) && return # return if the var has zero degree of freedom
    idx1 = var.offset + rand(config.rng, 1:currdof[vi]) # randomly choose one var to update
    idx2 = var.offset + rand(config.rng, 1:currdof[vi]) # randomly choose one var to update
    (idx1 == idx2) && return

    # oldvar = copy(var[idx])
    # currAbsWeight = config.absWeight
    currProbability = config.probability

    prop = Dist.swap!(var, idx1, idx2, config)

    # sampler may want to reject, then prop has already been set to zero
    if prop <= eps(0.0)
        return
    end

    weight = (fieldcount(V) == 1) ? integrand(curr, config.var[1], config) : integrand(curr, config.var, config)
    # weight = integrand_wrap(curr, config, integrand)
    newProbability = abs(weight) * config.reweight[curr]
    R = prop * newProbability / currProbability
    # newAbsWeight = abs(weight)
    # currAbsWeight = config.absWeight
    # R = prop * newAbsWeight / currAbsWeight

    # curr == 2 && println("propose, $curr: old: $oldvar --> new: $(var[idx]), with R $newAbsWeight / $currAbsWeight * $prop = $R")
    config.propose[2, curr, vi] += 1.0
    if rand(config.rng) < R
        # curr == 2 && println("accept, $curr")
        config.accept[2, curr, vi] += 1.0
        config.weights[curr] = weight
        config.probability = newProbability
        # config.absWeight = newAbsWeight
        # setweight!(config, weight)
    else
        # var[idx] = oldvar
        # config.absWeight = currAbsWeight
        Dist.swapRollback!(var, idx1, idx2, config)
    end
    return
end