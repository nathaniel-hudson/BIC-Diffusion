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


def __fast_LAIM(model, n_seeds, influencers, influencees, psi, max_iter=2, theta=0.0001):
    if n_seeds < 1:
        return set()

    max_iter += 1
    influence = [[0 for j in range(max_iter + 1)] for i in model.graph.nodes()]
    for node in model.graph.nodes():
        influence[node][1] = (1 - model.init_opinion[node])

    for i in range(max_iter):
        for u in model.graph.nodes():
            for v in influencees[u]:
                pp_uv = model.prop_prob(u, v, use_attempts=False)
                pp_vu = model.prop_prob(v, u, use_attempts=False) 
                influence[u][i] += pp_uv * (influence[v][i-1] - pp_vu * influence[u][i-2])
                # influence[u][i] -= (1 - pp_uv) * (1 - psi[v])
            influence[u][max_iter] += influence[u][i]

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

            ## NOTE: Penalty-free version.
            if influence[candidate][max_iter] > max_val:
                max_val  = influence[candidate][max_iter] 
                new_seed = candidate
        seeds.add(new_seed)
        covered_influencees = covered_influencees.union(temp_influencees)
        
    return seeds


def new_solution(model, n_seeds, p=0.01, theta=1.0):
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
    for node in graph.nodes():
        # if len(influencers[node]) == 0:
        #     inf_opinion_std[node] = 0
        #     continue
        opinions = [opinion[influencer] for influencer in influencers[node]]
        inf_opinion_std[node] = np.std(opinions + [opinion[node]])
    
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

    return __fast_LAIM(model, n_seeds, influencers, influencees, resilience)
