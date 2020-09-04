import networkx as nx
import numpy as np

def proposed_solution(model, n_seeds, max_iters=2):
    max_iters += 1
    graph = model.graph
    node_set = set(graph.nodes())
    
    impact = np.zeros(shape=(max_iters, len(node_set), len((node_set))))
    for u in node_set:
        impact[1][u][u] = 1.0

    for itr in range(max_iters):
        for u in node_set:
            for v in graph.neighbors(u):
                pp_uv = model.prop_prob(u, v, use_attempts=False)
                pp_vu = model.prop_prob(v, u, use_attempts=False)
                impact[itr][u][v] += pp_uv * \
                    (impact[itr-1][v][v] - pp_vu * impact[itr-2][u][u])
                # influence[u][i] += pp_uv * \
                    # (influence[v][i-1] - pp_vu * influence[u][i-2])
            impact[max_iters][u][u] += impact[itr][u][u]

    diagonal = impact[max_iters].diagonal()
    seed_set = set()
    for k in range(n_seeds):
        new_seed = np.argmax(diagonal)
        seed_set.add(new_seed)
        diagonal[new_seed] = float("-inf")
    
    return seed_set
