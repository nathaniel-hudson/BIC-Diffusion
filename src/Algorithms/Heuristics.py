import random 

from heapq import nlargest

def degree_solution(model, n_seeds):
    """Select the top subset of nodes ranked in decreasing order by (out-)degree.

    Parameters
    ----------
    graph : BIC_Model
        Instance of *prepared* BIC model.
    n_seeds : int
        Number of nodes to select as seeds/initial spreaders.

    Returns
    -------
    set
        Set of node IDs that were selected to be seeds.
    """
    score = {}
    for node in model.graph.nodes():
        score[node] = model.graph.degree(node)
    return set(nlargest(n_seeds, score, key=score.get))


def IRIE_solution(model, n_seeds, alpha=0.7):
    """Influence maximization solution proposed by Kyomin Jung, Wooram Heo, and Wei Chen 
       in "IRIE: Scalable and Robust Influence Maximization in Social Networks" in 2011.

    Parameters
    ----------
    model : BIC_Model
        Instance of the BIC model with the graph and respective data.
    n_seeds : int
        Number of nodes to be seeded.
    alpha : float, optional
        Hyperparameter for damping factor by suggestion of author's, by default 0.7.

    Returns
    -------
    set
        Set of seed nodes.
    """
    seeds = set()
    rank  = {node: 1 for node in model.graph.nodes()}
    AP_S  = {node: 0 for node in model.graph.nodes()}

    while len(seeds) < n_seeds:
        for node in model.graph.nodes():
            if node in seeds:
                AP_S[node] = 1
            else:
                AP_S[node] = 0
                for seed in seeds:
                    if node in list(model.graph.neighbors(seed)):
                        AP_S[node] += model.prop_prob(seed, node, use_attempts=False)

        for i in range(5):
            for u in model.graph.nodes():
                rank[u] = (1 - AP_S[u]) * (1 + alpha * sum([
                        model.prop_prob(u, v, use_attempts=False) * rank[v]
                        for v in model.graph.neighbors(u)
                    ]))

        num_of_seeds_before_new_seed = len(seeds)
        new_seed = max(rank, key=rank.get)
        seeds = seeds.union({new_seed})

        # Sometimes, IRIE estimates that the value in adding any node is nonexistant. So,
        # all the rank values will be equal to 0. This may lead to an infinite loop because
        # no new seeds are being selected from the above code (i.e., `new_seed` was already
        # a seed). So, to correct for this, we make a copy of `rank` where seeds' values
        # are -inf to avoid this issue.
        if len(seeds) == num_of_seeds_before_new_seed:
            rank_copy = rank.copy()
            for seed in seeds:
                rank_copy[seed] = float("-inf")
            new_seed = max(rank_copy, key=rank_copy.get)
            seeds = seeds.union({new_seed})

    return seeds


def min_opinion_solution(model, n_seeds):
    """Select the chosen number of nodes with minimal opinion in the OSN.

    Parameters
    ----------
    model : BIC_Model
        Instance of the BIC model with the graph and respective data.
    n_seeds : int
        Number of nodes to be seeded.

    Returns
    -------
    set
        Set of seed nodes.
    """
    ranker = lambda node: model.init_opinion[node]
    return set(sorted(list(model.graph.nodes()), key=ranker)[:n_seeds])


def random_solution(model, n_seeds):
    """Select a random subset of nodes for seeding.

    Parameters
    ----------
    graph : BIC_Model
        Instance of *prepared* BIC model.
    n_seeds : int
        Number of nodes to select as seeds/initial spreaders.

    Returns
    -------
    set
        Set of node IDs that were selected to be seeds.
    """
    return set(random.sample(model.graph.nodes, n_seeds))
