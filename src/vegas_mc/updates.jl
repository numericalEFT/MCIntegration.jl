# function changeIntegrand(config, integrand)
#     # update to change an integrand to its neighbors. 
#     # The degrees of freedom could be increase, decrease or remain the same.

#     curr = config.curr
#     new = rand(config.rng, config.neighbor[curr]) # jump to a randomly picked neighboring integrand

#     currProbability = config.probability

#     # propose probability caused by the selection of neighbors
#     prop = length(config.neighbor[curr]) / length(config.neighbor[new])

#     # sampler may want to reject, then prop has already been set to zero
#     if prop <= eps(0.0)
#         return
#     end

#     prop *= Dist.delta_probability(config, curr; new=new)

#     ############## the following are consistent  #############################
#     # prop *= Dist.delta_probability(config, curr; new=new)
#     # p1 = Dist.probability(config, curr)
#     # p2 = Dist.probability(config, new)
#     # @assert abs(p1 / p2 - prop) < 1e-10 "$p1   $p2   $prop"
#     ######################################################################

#     weights = integrand_wrap(config, integrand)
#     # newProbability = (new == config.norm) ? config.reweight[new] : abs(weights[new]) / Dist.probability(config, new) * config.reweight[new]
#     newProbability = (new == config.norm) ? config.reweight[new] : abs(weights[new]) * config.reweight[new]
#     # R = prop * newAbsWeight * config.reweight[new] / currAbsWeight / config.reweight[curr]
#     R = prop * newProbability / currProbability

#     config.propose[1, curr, new] += 1.0
#     if rand(config.rng) < R  # accept the change
#         config.accept[1, curr, new] += 1.0
#         setWeight!(config, weights)
#         config.probability = newProbability
#         config.curr = new
#         # else # reject the change
#         #     config.curr = curr # reset the current diagram index
#     end
#     return
# end

function changeVariable(config::Configuration{N,V,P,O,T}, integrand,
    currProbability::Float64, weights,
    padding_probability, _padding_probability) where {N,V,P,O,T}
    # update to change the variables of the current diagrams
    maxdof = config.maxdof
    vi = rand(config.rng, 1:length(maxdof)) # update the variable type of the index vi
    var = config.var[vi]
    if (var isa Discrete) && (var.size == 1) # there is only one discrete element, there is nothing to sample with.
        return currProbability
    end
    if maxdof[vi] <= 0
        return currProbability
    end
    idx = var.offset + rand(config.rng, 1:maxdof[vi]) # randomly choose one var to update

    prop = Dist.shift!(var, idx, config)

    # sampler may want to reject, then prop has already been set to zero
    if prop <= eps(0.0)
        return currProbability
    end

    # weights = integrand_wrap(config, integrand)
    _weights = (fieldcount(V) == 1) ?
               integrand(config.var[1], config) :
               integrand(config.var, config)
    # evaulation acutally happens before this step
    config.neval += 1

    for i in 1:N+1
        _padding_probability[i] = Dist.padding_probability(config, i)
    end
    # println(_padding_probability)
    newProbability = config.reweight[config.norm] * _padding_probability[config.norm] #normalization integral
    for i in 1:N #other integrals
        newProbability += abs(_weights[i]) * config.reweight[i] * _padding_probability[i]
    end
    R = prop * newProbability / currProbability

    config.propose[2, 1, vi] += 1.0
    if rand(config.rng) < R
        config.accept[2, 1, vi] += 1.0
        for i in 1:N # broadcast operator . doesn't work here, because _weights can be a scalar
            weights[i] = _weights[i]
        end
        for i in 1:N+1 # broadcast operator . doesn't work here, because _weights can be a scalar
            padding_probability[i] = _padding_probability[i]
        end
        # config.probability = newProbability
        return newProbability
    else
        Dist.shiftRollback!(var, idx, config)
        return currProbability
    end
    # return
end

# function swapVariable(config, integrand)
#     # update to change the variables of the current diagrams
#     curr = config.curr
#     maxdof = config.maxdof
#     vi = rand(config.rng, 1:length(maxdof)) # update the variable type of the index vi
#     var = config.var[vi]
#     idx1 = var.offset + rand(config.rng, 1:maxdof[vi]) # randomly choose one var to update
#     idx2 = var.offset + rand(config.rng, 1:maxdof[vi]) # randomly choose one var to update
#     (idx1 == idx2) && return

#     currProbability = config.probability

#     prop = Dist.swap!(var, idx1, idx2, config)

#     # sampler may want to reject, then prop has already been set to zero
#     if prop <= eps(0.0)
#         return
#     end


#     weights = integrand(config)
#     # newProbability = (curr == config.norm) ? config.reweight[curr] : abs(weights[curr]) / Dist.probability(config, curr) * config.reweight[curr]
#     newProbability = (curr == config.norm) ? config.reweight[curr] : abs(weights[curr]) * config.reweight[curr]

#     if (idx1 > config.dof[curr][vi] + var.offset) && (idx2 > config.dof[curr][vi] + var.offset)
#         R = 1.0
#     else
#         R = prop * newProbability / currProbability
#     end
#     R = prop * newProbability / currProbability

#     config.propose[2, curr, vi] += 1.0
#     if rand(config.rng) < R
#         config.accept[2, curr, vi] += 1.0
#         setWeight!(config, weights)
#     else
#         Dist.swapRollback!(var, idx1, idx2, config)
#     end
#     return
# end