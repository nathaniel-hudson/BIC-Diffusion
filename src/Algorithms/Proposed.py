import networkx as nx
import numpy as np

def opinion_degree_solution(model, n_seeds):
    """Proposed algorithm that ranks nodes for seeding by weighting the degree of each
       node by its some retractive measure of "net gain" in total opinion. Considerations
       for a node's in-neighborhood are made when ranking individual nodes.

    Parameters
    ----------
    model : BIC_Model
        Instance of BIC model with OSN topology and other data.
    n_seeds : int
        Number of nodes to seed.

    Returns
    -------
    set
        Nodes selected to be seeds.
    """
    delta = {}
    opinion = model.init_opinion

    for node in model.graph.nodes():
        # Initialize this node's impact score and get its out-neighbors.
        impact = 0
        out_neighbors = list(model.graph.neighbors(node))
        # out_neighbors = list(model.graph[node].keys())

        # Weight each neighbor by the average net gain in opinion.
        for neighbor in out_neighbors:
            # TODO: Maybe tune this so that it works with the pp values instead?
            opinion_impact = 1 - (opinion[node] + opinion[neighbor])/2
            behavio_impact = model.behavioral_inf(neighbor)
            impact += opinion_impact * behavio_impact

        # Anneal the "impact" score of a node based on net gain in terms of opinion from
        # activating `node`. Then add that node's impact value to `delta`.
        impact *= (1 - opinion[node])
        delta[node] = impact

    S = set()
    for i in range(n_seeds):

        arg = max(delta, key=delta.get)
        S   = S.union({arg})
        delta[arg] = float('-inf')
        in_neighbors = list(model.graph.neighbors(arg)) \
                       if not nx.is_directed(model.graph) \
                       else list(model.graph.predecessors(arg))

        # Remove the impact of `arg` from each of `arg`s in-neighbors cumulative impact score.
        for neighbor in in_neighbors:
            opinion_impact = 1 - (opinion[node] + opinion[neighbor])/2
            behavio_impact = model.behavioral_inf(arg)
            delta[neighbor] -= opinion_impact * behavio_impact

    return S


def __approx_influence(model, seeded, influencers, influencees, psi, inf_opinion_avg, max_iter=2, theta=0.0001):
    node_set = set(model.graph.nodes()) - seeded
    max_iter += 1
    influence = [[0 for j in range(max_iter + 1)] for i in model.graph.nodes()]
    cost = [[0 for j in range(max_iter + 1)] for i in model.graph.nodes()]
    for node in node_set:
        influence[node][1] = (1 - model.init_opinion[node])
        cost[node][1] = (inf_opinion_avg[node] - model.init_opinion[node])

    for i in range(max_iter):
        for u in node_set:
            for v in influencees[u]:
                pp_uv = model.prop_prob(u, v, use_attempts=False)
                pp_vu = model.prop_prob(v, u, use_attempts=False) 
                influence[u][i] += pp_uv * (influence[v][i-1] - pp_vu * influence[u][i-2])
                cost[u][i] += (1-pp_uv) * (cost[v][i-1] - (1-pp_vu) * cost[u][i-2]) ## NOTE: This is bad...
            influence[u][max_iter] += influence[u][i]
            cost[u][max_iter] += cost[u][i]

    seed = None
    max_val = float("-inf")
    for cand in node_set:
        val = influence[cand][max_iter] + cost[cand][max_iter]
        if val > max_val:
            seed = cand
            max_val = val
    return seed


def __LAIM(model, n_seeds, influencers, influencees, psi, inf_opinion_avg, max_iter=2, theta=0.0001):
    seeds = set()
    node_set = set(model.graph.nodes())
    for _ in range(n_seeds):
        new_seed = __approx_influence(model, seeds, influencers, influencees, psi, inf_opinion_avg, max_iter, theta)
        seeds.add(new_seed)
    return seeds



def __fast_LAIM(model, n_seeds, influencers, influencees, psi, inf_opinion_avg, max_iter=2, theta=0.0001):
    if n_seeds < 1:
        return set()

    max_iter += 1
    influence = [[0 for j in range(max_iter + 1)] for i in model.graph.nodes()]
    cost = [[0 for j in range(max_iter + 1)] for i in model.graph.nodes()]
    for node in model.graph.nodes():
        influence[node][1] = (1 - model.init_opinion[node])
        cost[node][1] = (inf_opinion_avg[node] - model.init_opinion[node])

    for i in range(max_iter):
        for u in model.graph.nodes():
            for v in influencees[u]:
                pp_uv = model.prop_prob(u, v, use_attempts=False)
                pp_vu = model.prop_prob(v, u, use_attempts=False) 
                influence[u][i] += pp_uv * (influence[v][i-1] - pp_vu * influence[u][i-2])
                cost[u][i] += (1-pp_uv) * (cost[v][i-1] - (1-pp_vu) * cost[u][i-2]) ## NOTE: This is bad...
            influence[u][max_iter] += influence[u][i]
            cost[u][max_iter] += cost[u][i]

    seeds = set()
    node_set = set(model.graph.nodes())
    covered_influencees = set()
    for _ in range(n_seeds):
        max_val = float("-inf")
        new_seed = None
        temp_influencees = set()
        for candidate in node_set - seeds:
            ## NOTE: Penalty version.
            # cand_influencees = influencees[candidate]
            # num = len(cand_influencees.intersection(covered_influencees))
            # den = len(cand_influencees.union(covered_influencees))
            # if den == 0:
            #     continue
            # val = influence[candidate][max_iter] * (num/den)
            # if val > max_val:
            #     max_val  = val
            #     new_seed = candidate
            #     temp_influencees = cand_influencees

            ## NOTE: Influence * Cost approach.
            val = influence[candidate][max_iter] + cost[candidate][max_iter]
            if val > max_val:
                max_val = val
                new_seed = candidate

            ## NOTE: Penalty-free version.
            # if influence[candidate][max_iter] > max_val:
            #     max_val  = influence[candidate][max_iter] 
            #     new_seed = candidate
        seeds.add(new_seed)
        covered_influencees = covered_influencees.union(temp_influencees)
        
    return seeds


def new_solution222(model, n_seeds, p=0.01, theta=1.0):
    """Proposed solution.

    Parameters
    ----------
    model : BIC
        Instance of the BIC model.
    n_seeds : int
        Number of seed nodes.
    p : float, optional
        Discount tuning parameter, by default 0.01.

    Returns
    -------
    set
        Set of seed nodes.
    """
    graph = model.graph
    opinion = model.init_opinion
    get_in_neighbors = graph.neigbors if not nx.is_directed(graph) else graph.predecessors
    influencers = {node: set(get_in_neighbors(node)) for node in graph.nodes()}
    influencees = {node: set(graph.neighbors(node))  for node in graph.nodes()}
    inf_opinion_std = {}
    inf_opinion_avg = {}
    for node in graph.nodes():
        # if len(influencers[node]) == 0:
        #     inf_opinion_std[node] = 0
        #     continue
        opinions = [opinion[influencer] for influencer in influencers[node]]
        inf_opinion_std[node] = np.std(opinions + [opinion[node]])
        inf_opinion_avg[node] = np.mean(opinions + [opinion[node]])
    
    def general_influencers(v):
        min_val = opinion[v] - inf_opinion_std[v]
        max_val = opinion[v] + inf_opinion_std[v]
        return set([
            u for u in influencers[v] 
            if min_val <= opinion[u] and opinion[u] <= max_val
        ])

    def penalized_influencers(v):
        max_val = opinion[v] + inf_opinion_std[v]
        return set([
            u for u in influencers[v] 
            if opinion[u] <= max_val
        ])

    resilience = {}
    for node in graph.nodes():
        gen_influencers = general_influencers(node)
        pen_influencers = penalized_influencers(node)
        if len(pen_influencers) == 0:
            resilience[node] = 1.0
        else:
            resilience[node] = len(gen_influencers) / len(pen_influencers)

    return __LAIM(model, n_seeds, influencers, influencees, resilience, inf_opinion_avg)
    # return __fast_LAIM(model, n_seeds, influencers, influencees, resilience, inf_opinion_avg)


def new_solution(model, n_seeds, max_iters=2):
    max_iters += 1
    graph = model.graph
    node_set = set(graph.nodes())
    opinion = model.init_opinion.copy()

    in_neighbors = graph.neighbors if not nx.is_directed(graph) else graph.predecessors
    out_neighbors = graph.neighbors

    def pen_influencers(node):
        influencers = set()
        in_neighbors_opinions = [opinion[x] for x in in_neighbors(node)]
        in_neighbors_opinions.append(opinion[node])
        std = np.std(in_neighbors_opinions)

        for in_nbor in in_neighbors(node):
            if opinion[in_nbor] <= opinion[node] + std:
                influencers.add(in_nbor)
        
        return influencers
    
    impact = np.zeros(shape=(max_iters+1, len(node_set), len(node_set)))
    penalty = np.zeros(shape=(len(node_set), len(node_set)))
    for v in node_set:
        impact[1][v][v] = 1.0 - opinion[v]
        
        influencers = pen_influencers(v)
        for u in influencers:
            diff = (opinion[u] - opinion[v]) / len(influencers)
            penalty[u][v] += diff
            penalty[v][v] += diff

    for itr in range(2, max_iters):
        for u in node_set:
            for v in out_neighbors(u):
                pp_uv = model.prop_prob(u, v, use_attempts=False)
                pp_vu = model.prop_prob(v, u, use_attempts=False)

                impact[itr][u][v] += pp_uv * \
                    (impact[itr-1][v][v] - pp_vu * impact[itr-2][u][u])
                impact[itr][u][u] += impact[itr][u][v]

                ## NOTE: Experimental.
                # penalty[itr][u][v] += (1-pp_uv) * \
                #     penalty[itr][v][v]
                #     # (penalty[itr][v][v] - (1-pp_vu) * penalty[itr-2][u][u])
                # penalty[itr][u][u] += impact[itr][u][v]

            impact[max_iters][u][u] += impact[itr][u][u]
            # impact[max_iters][u][u] = impact[1][u][u] + impact[max_iters][u][u] + impact[itr][u][u]

            ## NOTE: Experimental.
            # penalty[max_iters][u][u] = penalty[1][u][u] + penalty[max_iters][u][u] + penalty[itr][u][u]
            

    seed_set = set()
    for k in range(n_seeds):
        v = None
        max_val = float("-inf")

        for cand in node_set - seed_set:
            val = impact[max_iters][cand][cand] + penalty[cand][cand]
            if val > max_val:
                v = cand
                max_val = val
        seed_set.add(v)

        # Anneal the `impact` matrix based on the newly seeded node.
        for u in in_neighbors(v):
            impact[max_iters][u][u] -= impact[max_iters-1][u][v]
            impact[max_iters][u][u] -= impact[max_iters-1][v][u]
            
        opinion[v] = 1.0
        penalty = np.zeros(shape=(len(node_set), len(node_set)))
        for z in node_set:
            influencers = pen_influencers(z)
            for y in influencers:
                diff = (opinion[y] - opinion[z]) / len(influencers)
                penalty[y][z] += diff
                penalty[z][z] += diff

    return seed_set
