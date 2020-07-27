import networkx as nx
import numpy as np

def revised_solution(model, n_seeds):
    ## gain(pp) +- loss(1 - pp)
    return set()

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


def my_floydw(model):
    """Modified version of the Floyd-Warshall algorithm, that finds the "likeliest" paths.
    
    Parameters
    ----------
    model : BIC_Model
        Instance of BIC model with OSN topology and other data.
    
    Returns
    -------
    np.array
        2D matrix containing probability for v to be activated as a result of node u.
    """
    dist = np.zeros((len(model.graph), len(model.graph)))
    if nx.is_directed(model.graph):
        for (u,v) in model.graph.edges():
            dist[u][v] = model.prop_prob(u, v, 0, use_attempts=False)
    else:
        for (u,v) in model.graph.edges():
            dist[u][v] = model.prop_prob(u, v, 0, use_attempts=False)
            dist[u][v] = model.prop_prob(v, u, 0, use_attempts=False)
    
    for v in model.graph.nodes():
        dist[v][v] = 1.0
    
    for k in model.graph.nodes():
        for i in model.graph.nodes():
            for j in model.graph.nodes():
                if dist[i][j] < dist[i][k] * dist[k][j]:
                    dist[i][j] = dist[i][k] * dist[k][j]
    return dist


def opinion_path_solution(model, n_seeds):
    """An older proposed algorithm for seeding nodes by computing paths of probability for activation.

    Parameters
    ----------
    model : BIC_Model
        Instance of BIC model.
    n_seeds : int
        Number of nodes to seed.

    Returns
    -------
    set
        Nodes selected to be seeds.
    """
    V = model.graph.nodes()
    dist = my_floydw(model)
    net_gain = [[1 - model.init_opinion[v]] for v in V]
    S = set()
    option = 1

    cpp_matrix = np.asarray(dist)
    sum_matrix = np.array([[sum(cpp_matrix[i])] for i in range(len(cpp_matrix))])
    net_gain = np.asarray(net_gain)
    T = [1 for v in range(len(V))]

    # ORIGINAL:
    # R = np.matmul(cpp_matrix, net_gain)

    best_fit = np.dot(cpp_matrix, sum_matrix) * net_gain

    for i in range(n_seeds):
        # Grab the most "influential" node based on the (delta * phi) computation.
        arg = np.argmax(best_fit)
        S = S.union({arg})

        ### NEW VERSION.
        # NOTE: I need to find a way to consider the in- and out-neighbors of `arg`.
        for j in range(len(best_fit)):
            cpp_matrix[arg][j] = cpp_matrix[j][arg] = 0.0

        best_fit = np.dot(cpp_matrix, sum_matrix) * net_gain

        '''
        This part needs to be played around with... HOW do we update it?
        '''
        '''
        ### Original version.
        for neighbor in model.graph.neighbors(arg):
            T[neighbor] += 1

        # "Delete" the influence the `arg`-th row and colulmn in `R`.
        delta[arg, :] = 0  # Fill the arg-th row in delta with 0.
        delta[:, arg] = 0  # Fill the arg-th col in delta with 0.
        R = np.matmul(delta, phi)

        # Create the neighborhood discount matrix.
        _T = np.array([[1/t] for t in T])
        R = R * _T
        '''

    return list(S)