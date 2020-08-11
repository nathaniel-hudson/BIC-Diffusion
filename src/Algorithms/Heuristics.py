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
