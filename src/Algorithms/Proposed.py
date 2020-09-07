import copy
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


def new_solution(model, n_seeds, max_iters=2, node_set=None):
    max_iters += 1
    graph = model.graph
    if node_set is None:
        node_set = set(graph.nodes())
    opinion = model.init_opinion.copy()

    in_neighbors = graph.neighbors if not nx.is_directed(graph) else graph.predecessors
    out_neighbors = graph.neighbors

    def gen_influencers(node):
        influencers = set()
        in_neighbors_opinions = [opinion[x] for x in in_neighbors(node)]
        in_neighbors_opinions.append(opinion[node])
        std = np.std(in_neighbors_opinions)

        for in_nbor in in_neighbors(node):
            if opinion[node]-std <= opinion[in_nbor] and opinion[in_nbor] <= opinion[node] + std:
                influencers.add(in_nbor)
        
        return influencers

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

            impact[max_iters][u][u] += impact[itr][u][u]
            

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

        # Reconstruct the penalty matrix.
        opinion[v] = 1.0
        penalty = np.zeros(shape=(len(node_set), len(node_set)))
        for z in node_set:
            influencers = pen_influencers(z)
            for y in influencers:
                diff = (opinion[y] - opinion[z]) / len(influencers)
                penalty[y][z] += diff
                penalty[z][z] += diff

    return seed_set
