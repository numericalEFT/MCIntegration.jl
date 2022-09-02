function changeIntegrand(config, integrand, userdata)
    # update to change an integrand to its neighbors. 
    # The degrees of freedom could be increase, decrease or remain the same.

    curr = config.curr
    new = rand(config.rng, config.neighbor[curr]) # jump to a randomly picked neighboring integrand

    currdof, newdof = config.dof[curr], config.dof[new]

    currAbsWeight = config.absWeight

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

    config.curr = new
    weight = (new == config.norm ? 1.0 : integrand_wrap(config, integrand, userdata))
    newAbsWeight = abs(weight)
    R = prop * newAbsWeight * config.reweight[new] / currAbsWeight / config.reweight[curr]

    config.propose[1, curr, new] += 1.0
    if rand(config.rng) < R  # accept the change
        config.accept[1, curr, new] += 1.0
        config.absWeight = newAbsWeight
        setweight!(config, weight)
    else # reject the change
        config.curr = curr # reset the current diagram index
        # config.absWeight = currAbsWeight

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

function changeVariable(config, integrand, userdata)
    # update to change the variables of the current diagrams
    (config.curr == config.norm) && return

    curr = config.curr
    currdof = config.dof[curr]
    vi = rand(config.rng, 1:length(currdof)) # update the variable type of the index vi
    var = config.var[vi]
    (currdof[vi] <= 0) && return # return if the var has zero degree of freedom
    idx = var.offset + rand(config.rng, 1:currdof[vi]) # randomly choose one var to update

    # oldvar = copy(var[idx])
    currAbsWeight = config.absWeight

    prop = Dist.shift!(var, idx, config)

    # sampler may want to reject, then prop has already been set to zero
    if prop <= eps(0.0)
        return
    end

    weight = integrand_wrap(config, integrand, userdata)
    newAbsWeight = abs(weight)
    currAbsWeight = config.absWeight
    R = prop * newAbsWeight / currAbsWeight

    # curr == 2 && println("propose, $curr: old: $oldvar --> new: $(var[idx]), with R $newAbsWeight / $currAbsWeight * $prop = $R")
    config.propose[2, curr, vi] += 1.0
    if rand(config.rng) < R
        # curr == 2 && println("accept, $curr")
        config.accept[2, curr, vi] += 1.0
        config.absWeight = newAbsWeight
        config.relativeWeight = weight / newAbsWeight / config.reweight[config.curr]
        # setweight!(config, weight)
        # accumulate!(var, idx)
    else
        # var[idx] = oldvar
        # config.absWeight = currAbsWeight
        Dist.shiftRollback!(var, idx, config)
    end
    return
end

function swapVariable(config, integrand, userdata)
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
    currAbsWeight = config.absWeight

    prop = Dist.swap!(var, idx1, idx2, config)

    # sampler may want to reject, then prop has already been set to zero
    if prop <= eps(0.0)
        return
    end

    weight = integrand_wrap(config, integrand, userdata)
    newAbsWeight = abs(weight)
    currAbsWeight = config.absWeight
    R = prop * newAbsWeight / currAbsWeight

    # curr == 2 && println("propose, $curr: old: $oldvar --> new: $(var[idx]), with R $newAbsWeight / $currAbsWeight * $prop = $R")
    config.propose[2, curr, vi] += 1.0
    if rand(config.rng) < R
        # curr == 2 && println("accept, $curr")
        config.accept[2, curr, vi] += 1.0
        config.absWeight = newAbsWeight
        setweight!(config, weight)
    else
        # var[idx] = oldvar
        # config.absWeight = currAbsWeight
        Dist.swapRollback!(var, idx1, idx2, config)
    end
    return
end