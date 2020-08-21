import itertools
import networkx as nx
import numpy as np
import operator
import pandas as pd
import random
import time

from heapq import nlargest
from itertools import count

# Model-based imports.
import BIC

ROUNDS = 100 # 1000 #50
HORIZON = 100


def celf(g, k, opinion, ffm, rounds=ROUNDS):
    """Implementation of the CELF algorithm proposed by Leskovec et al. 
    
    Arguments:
        g {nx.Graph} -- a NetworkX Graph object.
        k {int} -- the number of seed nodes to return.
    
    Keyword Arguments:
        rounds {int} -- number of simulation runs for each seed k-permutations. (default: ROUNDS)
    
    Returns:
        set -- Set of nodes to be used as seeds, s.t. len(S) == k.
    """

    def make_g_prime(g):
        """Support function that produces a simulation graph which only contains 'live'
           edges using intiial activation probabilities to decide live/dead edges.
        
        Arguments:
            g {nx.Graph} -- Graph representing the OSN under consideration.
        
        Returns:
            nx.DiGraph -- Resulting simulation graph of only live (activated) edges.
        """
        g_prime = nk.graph.Graph(directed=True)
        g_prime.add_nodes_from(g.nodes)
        for (u, v) in g.edges():
            # NOTE: (graph, source, target, attempt, opinion, ffm, t)
            if random.random() <= BIC.prop_prob(g, u, v, 1, opinion, ffm, 0):
                g_prime.addEdge(u, v)
        return g_prime

    def reachable(g, sources):
        """Support function that simply returns a list of all the nodes that are reachable 
           from the provided source nodes.
        
        Arguments:
            g {nx.Graph} -- Graph representing the OSN under consideration.
            sources {list} -- Nodes that we use as sources to find the "reachable" nodes.
        
        Returns:
            list -- List of all the reachable nodes from the nodes in `sources`.
        """
        reachable_nodes = []
        for source in sources:
            for target in g.nodes:
                if source == target:
                    continue
                if nx.has_path(g, source, target):
                    reachable_nodes.append(target)
        return reachable_nodes

    S = set()
    V = set(g.nodes)

    for i in range(k):
        sigma = [0 for node in g.nodes]
        for j in range(rounds):
            live_graph = make_g_prime(g)
            R = {}
            R[tuple(S)] = reachable(live_graph, S)
            for v in V:
                R[v] = reachable(live_graph, [v])

            for v in V - S:
                if v not in R[tuple(S)]:
                    sigma[v] += len(R[v])

        for v in V:
            if v in S:
                sigma[v] = float('-inf')
            else:
                sigma[v] /= rounds

        _sigma = np.array(sigma)
        S = S.union({np.argmax(_sigma)})

    return S
    
'''
def MATI_IC(g, k, attr='opinion'):

    def calc_stats_IC(g):
        A = {}
        for u in g.nodes:
            # TODO: Generate T(u) and F(tau_i), forall tau_i using DFS
            for v in g.nodes:
                pr = 1
                # TODO: Generate Psi(u, v) and phi(psi_i), forall psi_i (based on T(u))
                for psi in nx.all_simple_paths(u, v): #Psi(u, v):
                    pr = pr * (1 - phi(i, j))
                # end for
                A[u, v] = 1 - pr
            # end for
        # end for
        return A

    def calc_inf(A, V):
        Q = set()
        sigma = {u: 0 for u in V}

        for u in V:
            for v in I(u):
                sigma[u] += A(u,v)
            Q.append((u, sigma[u]))
        
        return Q

    def additive_inf(T, F, S):
        i, inf = 0, 0
        for t in T:
            i = i + 1
            for u in t:
                j = index(u)
                if j == 1:
                    continue
                elif u not in S:
                    inf += f(i, j)
                else:
                    break
        return inf

    def get_paths(graph):
        """Generates the set of possible paths from each node, as described by the MATI 
           paper by Rossi et al., such that no path is a subpath of another path. This
           function essentially generates a dictionary that fits the T(*) function for the
           model described in the original MATI paper.
        
        Arguments:
            graph {nx.GiGraph} -- Directed graph representing the considered OSN.
        
        Returns:
            dict -- Set of paths available to root nodes from each node in `graph`.
        """
        if nx.is_directed(graph) == False:
            graph = nx.to_directed(graph)

        paths = {}
        for node in graph.nodes():
            tree = nx.dfs_tree(graph, source=node)
            terminal_nodes = [x for x in tree.nodes() if graph.out_degree(x) == 0]
            paths[node] = []
            for term in terminal_nodes:
                paths[node] += [path for path in nx.all_simple_paths(graph, node, term)]
        return paths

    S = set()
    V = set(g.nodes)
    T = get_paths(g)
    A = calc_stats_IC(g)
    Q = calc_inf(A, V)
    sigma = {S: 0}

    # TODO: We need to review the MATI paper and fill in the rest of this code (i.e., im-
    #       plement I(*), T(*), and F(*)).
    # NOTE: I(u) -- the nodes influenced by u.
    #       T(u) -- set of all possible paths starting from node u.
    #       F(t) -- cumulative probability path for path t.
    for i in range(k):
        elt = Q.pop() # Will be a tuple with (node, inf) pair.
        s, sigma[elt[0]] = elt[0], elt[1]
        S = S.union({s})
        U = V - S
        for u in U:
            sigma[S.union({u})] = len(S.union({u}))
            sigma[S.union({u})] += additive_inf(T[u], F(u), S)
            sigma[S.union({u})] += additive_inf(T[u], F(u), S.union({u}))
            Q.append((u, sigma[S.union({u})] - sigma[S])) # TODO: Make it an ordered insert.
    
    return S
'''




def IRIE(g, k, opinion, ffm, alpha=0.7):
    """Implementation of the IRIE algorithm.
    
    Arguments:
        g {nx.Graph} -- a NetworkX Graph object.
        k {int} -- the number of seed nodes to return.
        opinion {dict} -- Nodewise opinion values.
        ffm {dict} -- Nodewise five-factor model parameters.
    
    Keyword Arguments:
        alpha {float} -- Tunable parameter. (default: {0.7})
    
    Returns:
        set -- Set of nodes to be used as seeds, s.t. len(S) == k.
    """
    S = set()
    V = set(g.nodes())
    rank = {1 for v in V}
    AP_S = {0 for v in V}

    while len(S) < k:
        for v in V:
            if v in S:
                AP_S[v] = 1
            else:
                # Estimate the AP value.
                AP_S[v] = 0
                for s in S:
                    if v in list(g.neighbors(s)):
                        # (graph, source, target, attempt, opinion, ffm, t)
                        AP_S[v] += BIC.prop_prob(g, s, v, 1, opinion, ffm, 0)
        
        for i in range(5):
            for u in V:
                rank[u] = (1 - AP_S[u]) * (1 + alpha * \
                    sum([BIC.prop_prob(g, u, v, 1, opinion, ffm, 0) * rank[v] for v in g.neighbors(u)]))

        temp = len(S)
        u = max(rank, key=rank.get)
        S = S.union({u})

        # Sometimes, IRIE estimates that the value in adding any node is nonexistent, so
        # all the rank values will be 0. This may lead to an infinite loop. So, invalidate
        # already selected seeds in a copy of `rank` where the seeds' values are -inf.
        if len(S) == temp:
            rank_copy = rank.copy()
            # Invalidate nodes already selected as seeds.
            for s in S:
                rank_copy[s] = float('-inf')
            u = max(rank_copy, key=rank_copy.get)
            S = S.union({u})

    return S


def OM_IRIE(g, k, opinion, ffm, alpha=0.7):
    """Implementation of the IRIE algorithm.
    
    Arguments:
        g {nx.Graph} -- a NetworkX Graph object.
        k {int} -- the number of seed nodes to return.
        opinion {dict} -- Nodewise opinion values.
        ffm {dict} -- Nodewise five-factor model parameters.
    
    Keyword Arguments:
        alpha {float} -- Tunable parameter. (default: {0.7})
    
    Returns:
        set -- Set of nodes to be used as seeds, s.t. len(S) == k.
    """
    S = set()
    V = set(g.nodes())
    rank = {}
    AP_S = {}
    
    for u in V:
        rank[u] = 1
        AP_S[u] = 0

    while len(S) < k:
        for v in V:
            if v in S:
                AP_S[v] = 1
            else:
                # Estimate the AP value.
                AP_S[v] = 0
                for s in S:
                    if v in list(g.neighbors(s)):
                        # (graph, source, target, attempt, opinion, ffm, t)
                        AP_S[v] += BIC.prop_prob(g, s, v, 1, opinion, ffm, 0) * (1 - opinion[v][0]) # NOTE: Added the multiplicative.
        
        for i in range(5):
            for u in V:
                rank[u] = (1 - AP_S[u]) * (1 + alpha * sum([BIC.prop_prob(g, u,
                    v, 1, opinion, ffm, 0) * rank[v] for v in g.neighbors(u)]))

        temp = len(S)
        u = max(rank, key=rank.get)
        S = S.union({u})

        # Sometimes, IRIE estimates that the value in adding any node is nonexistent, so
        # all the rank values will be 0. This may lead to an infinite loop. So, invalidate
        # already selected seeds in a copy of `rank` where the seeds' values are -inf.
        if len(S) == temp:
            rank_copy = rank.copy()
            # Invalidate nodes already selected as seeds.
            for s in S:
                rank_copy[s] = float('-inf')
            u = max(rank_copy, key=rank_copy.get)
            S = S.union({u})

    return S


def degree_discount(g, k, p=0.01):
    """ Efficient algorithmic solution to the IM problem proposed by Chen et al.

    Arguments:
        g {nx.Graph} -- a NetworkX Graph object.
        k {int} -- the number of seed nodes to return.

    Keyword Arguments:
        p {float} -- discount tuning parameter (default: 0.01).

    Returns:
        set -- Set of nodes to be used as seeds, s.t. len(S) == k.
    """
    if k <= 0:
        return set()

    V = set(g.nodes())
    S = set()
    DD = {node: g.degree(node) for node in V}
    T = {node: 0 for node in V}
    for i in range(k):
        u = None
        val = float('-inf')
        for node in V-S:
            if g.degree(node) > val:
                u = node
                val = DD[node]

        S = S.union({u})

        for v in g.neighbors(u):
            if v in V-S:
                T[v] += 1
                DD[v] = g.degree(v) - 2*T[v] - (g.degree(v)-T[v]) * T[v] * p

    return S


def greedy_OM(g, k, opinion, ffm, t_horizon=HORIZON, rounds=ROUNDS):
    """ Implementation of the Greedy Hill-Climbing Algorithm by Kempe et al. 
    
    Arguments:
        g {nx.Graph} -- a NetworkX Graph object.
        k {int} -- the number of seed nodes to return.

    Keyword Arguments:
        t_horizon {int} -- the number of discrete time-steps for each simulation run (default: 100).
        rounds {int} -- number of simulation runs for each seed k-permutations (default: 1000).

    Returns:
        set -- Set of nodes to be used as seeds, s.t. len(S) == k.
    """
    S = set()
    V = set(g.nodes())
    sigma = [0 for node in V]

    for i in range(k):
        for v in V - S:
            sigma[v] = 0
            for i in range(rounds):
                result = BIC.simulate(g, S.union({v}), opinion, ffm, t_horizon, ROUNDS)
                # sigma[v] += result['time-step'][t_horizon]
                result = pd.DataFrame.from_dict(result)
                sigma[v] += result.loc[result['time-step'] == t_horizon, 'opinion'].tolist()[0]
            sigma[v] = sigma[v]/rounds

        # Get the node with the highest sigma (influence).
        _sigma = np.array(sigma)
        S = S.union({np.argmax(_sigma)})

    return S


def greedy_IM(g, k, opinion, ffm, t_horizon=HORIZON, rounds=ROUNDS):
    """ Implementation of the Greedy Hill-Climbing Algorithm by Kempe et al. 
    
    Arguments:
        g {nx.Graph} -- a NetworkX Graph object.
        k {int} -- the number of seed nodes to return.
        t_horizon {int} -- the number of discrete time-steps for each simulation run (default=25).
        rounds {int} -- number of simulation runs for each seed k-permutations (default=25).

    
    Keyword Arguments:
        t_horizon {int} -- the number of discrete time-steps for each simulation run (default: 100).
        rounds {int} -- number of simulation runs for each seed k-permutations (default: 1000).

    Returns:
        set -- Set of nodes to be used as seeds, s.t. len(S) == k.
    """
    S = set()
    V = set(g.nodes)
    sigma = [0 for node in g.nodes]

    for i in range(k):
        for v in V - S:
            sigma[v] = 0
            for i in range(rounds):
                result = sim.run_single_activation(S, t_horizon, 1)
                sigma[v] += result.loc[result['time-step'] == t_horizon, 'activated'].tolist()[0]
            sigma[v] = sigma[v]/rounds

        # Get the node with the highest sigma (influence).
        _sigma = np.array(sigma)
        S = S.union({np.argmax(_sigma)})

    return S


def opinion_degree(g, n_seeds, opinion, ffm):
    """Heuristic algorithm that ranks nodes for seeding by weighting the degree of each
       node by its some retractive measure of "net gain" in total opinion. Considerations
       for a node's in-neighborhood are made when ranking individual nodes.
    
    Arguments:
        g {nx.Graph} -- Graph for the OSN under consideration.
        n_seeds {int} -- Seeding budget.
        opinion {dict} -- Nodewise opinion values.
        ffm {dict} -- Nodewise five-factor model parameters.
    
    Returns:
        set -- Set of nodes to be used as seeds, s.t. len(S) == k.
    """
    delta = {}
    for node in g.nodes():
        # Initialize this node's impact score and get its out-neighbors.
        impact = 0
        out_neighbors = list(g.neighbors(node))
        # out_neighbors = list(g[node].keys())

        # Weight each neighbor by the average net gain in opinion.
        for neighbor in out_neighbors:
            # TODO: Maybe tune this so that it works with the pp values instead?
            opinion_impact = 1 - (opinion[node][0] + opinion[neighbor][0])/2
            behavio_impact = BIC.behavioral_influence(g, neighbor, ffm)
            impact += opinion_impact * behavio_impact

        # Anneal the "impact" score of a node based on net gain in terms of opinion from
        # activating `node`. Then add that node's impact value to `delta`.
        impact *= (1 - opinion[node][0])
        delta[node] = impact

    S = set()
    for i in range(n_seeds):

        arg = max(delta, key=delta.get)
        S = S.union({arg})
        delta[arg] = float('-inf')
        in_neighbors = list(g.neighbors(arg)) if not nx.is_directed(g) else list(g.predecessors(arg))

        # Remove the impact of `arg` from each of `arg`s in-neighbors cumulative impact score.
        for neighbor in in_neighbors:
            opinion_impact = 1 - (opinion[node][0] + opinion[neighbor][0])/2
            behavio_impact = BIC.behavioral_influence(g, arg, ffm)
            delta[neighbor] -= opinion_impact * behavio_impact

    return S


def my_floydw(g, opinion, ffm):
    """Modified version of the Floyd-Warshall algorithm, that finds the "likeliest" paths.
    
    Arguments:
        g {nx.Graph} -- Graph representing the OSN under consideration.
        opinion {dict} -- Nodewise opinion values.
        ffm {dict} -- Nodewise five-factor model parameters.
    
    Returns:
        list<list> -- Two-dimensional matrix containing the probability for node v to be 
            activated as a result of node u being activated.
    """
    dist = [[0 for node in g.nodes()] for node in g.nodes()]
    if nx.is_directed(g):
        for (u,v) in g.edges():
            dist[u][v] = BIC.prop_prob(g, u, v, 1, opinion, ffm, 0)
    else:
        for (u,v) in g.edges():
            dist[u][v] = BIC.prop_prob(g, u, v, 1, opinion, ffm, 0)
            dist[v][u] = BIC.prop_prob(g, v, u, 1, opinion, ffm, 0)
    for v in g.nodes():
        dist[v][v] = 1.0
    for k in g.nodes():
        for i in g.nodes():
            for j in g.nodes():
                if dist[i][j] < dist[i][k] * dist[k][j]:
                    dist[i][j] = dist[i][k] * dist[k][j]
    return dist


def opinion_path(g, n_seeds, opinion, ffm):
    """ A naive, random solution that will serve as a baseline for comparison.
    
    Keyword arguments:
        g {nk.Graph} -- A NetworKit Graph object.
        k {int} -- Seeding budget.
        opinion {dict} -- Nodewise opinion values.
        ffm {dict} -- Nodewise five-factor model parameters.

    Returns:
        list -- Nodes to be used as seeds, s.t. len(S) == k.
    """
    V = g.nodes()
    dist = my_floydw(g, opinion, ffm)
    net_gain = [[1 - opinion[v][0]] for v in V]
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
        """
        ### Original version.
        for neighbor in g.neighbors(arg):
            T[neighbor] += 1

        # "Delete" the influence the `arg`-th row and colulmn in `R`.
        delta[arg, :] = 0  # Fill the arg-th row in delta with 0.
        delta[:, arg] = 0  # Fill the arg-th col in delta with 0.
        R = np.matmul(delta, phi)

        # Create the neighborhood discount matrix.
        _T = np.array([[1/t] for t in T])
        R = R * _T
        """

    return list(S)


# ====================================================================================== #
# ====================================================================================== #
# ====================================================================================== #

def harmonic(g, k):
    """Uses the Harmonic Centrality metric to rank nodes for seeding.
    
    Arguments:
        g {nx.Graph} -- Graph for OSN to consider.
        k {int} -- Seeding budget.
    
    Returns:
        list -- Nodes to be used as seeds, s.t. len(S) == k.
    """
    return nk.centrality.TopHarmonicCloseness(g, k).run().topkNodesList()


def degree(g, k):
    """Simple heuristic that selects the nodes with highest out-degree for seeding.
    
    Arguments:
        g {nx.Graph} -- Graph representing OSN under consideration.
        k {int} -- Seeding budget.

    Returns:
        list -- Nodes to be used as seeds, s.t. len(S) == k.
    """
    score = {}
    for node in g.nodes():
        score[node] = g.degree(node)
    return nlargest(k, score, key=score.get)


def min_opinion(g, k, opinion):
    """Generates a seed set of length `k` using a minimum opinion heuristic.
    
    Arguments:
        g {nx.Graph} -- Graph representing OSN under consideration.
        k {int} -- Seeding budget.
        opinion {dict} -- Nodewise opinion values.
    
    Returns:
        list -- Nodes to be used as seeds, s.t. len(S) == k.
    """
    return sorted(list(g.nodes()), key=lambda node: opinion[node][0])[:k]


def random_sol(g, k):
    """A naive, random solution that will serve as a baseline for comparison.
    
    Arguments:
        g {nx.Graph} -- Graph representing OSN under consideration.
        k {int} -- Seeding budget.

    Returns:
        list -- Nodes to be used as seeds, s.t. len(S) == k.
    """
    if k <= 0:
        return []
    return random.choices(range(g.number_of_nodes()), k=k)
