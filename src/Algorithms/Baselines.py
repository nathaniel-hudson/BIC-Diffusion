import networkx as nx
import math
import random

from .hypergraph import hypergraph

# from halp.undirected_hypergraph import UndirectedHypergraph

def IMM_solution(model, n_seeds, epsilon=0.0, ell=1.0):
    """Implementation of the LAIM algorithm proposed by Tang et al. in 2015 in the paper entitled "Influence 
       Maximization in Near-Linear Time: A Martingale Approach".

    Parameters
    ----------
    model : BIC_Model
        Instance of the BIC model.
    n_seeds : int
        Number of seed nodes.

    Returns
    -------
    set
        Nodes selected to be seeds.
    """
    ## NOTE: Currently, this is of least priority. The LAIM algorithm outperforms it, is more state-of-the-art, and we have
    ##       C++ code to translate into Python code to go off of.
    ell = ell * (1.0 + log(2)/log(len(model.graph)))
    R = sampling(model.graph, n_seeds, epsilon, ell)
    S = node_selection(R, n_seeds)
    return set()


def LAIM_solution(model, n_seeds, max_it=10, theta=0.1):
    """Implementation of the LAIM algorithm proposed by We et al. in 2018 in the paper entitled "LAIM: A Linear 
       Time Iterative Approach for Efficient Influence Maximization in Large-Scale Networks".

    Parameters
    ----------
    model : BIC_Model
        Instance of the BIC model.
    n_seeds : int
        Number of seed nodes.

    Returns
    -------
    set
        Nodes selected to be seeds.
    """
    ## Refer to C++ code provided in `./LAIM_cpp` for the code provided by the authors.
    ## NOTE: This needs a LOT of work.
    def calc_influence(gamma):
        I = {}
        for ind in range(-1, gamma+1):
            for u in model.graph.nodes:
                I[(u, ind)] = 0

        local_influence = {node: 0 for node in model.graph.nodes()}
        for ell in range(1, gamma+1):
            for u in model.graph.nodes:
                inf = sum(
                    model.prop_prob(u, v, 1) * (I[v, gamma-1] - model.prop_prob(u, v, 1) * I[u, gamma-2])
                    for v in model.graph.neighbors(u)
                )
                local_influence[u] += inf

    seeds = set()
    for i in range(n_seeds):
        influence = calc_influence()
        u = np.argmax(influence)
        influence[u] = -1
        seeds.add(u)

    return set()


def TIM_solution(model, n_seeds, time_horizon=100, epsilon=0.5):
    """Implementation of the TIM^{+} algorithm proposed by Borgs et al. in 2014 in the paper entitled "Maximizing Social 
       Influence in Nearly Optimal Time".

    Parameters
    ----------
    model : BIC_Model
        Instance of the BIC model.
    n_seeds : int
        Number of seed nodes.

    Returns
    -------
    set
        Nodes selected to be seeds.
    """

    def BIC_DFS(model, graph_transpose, start, R):
        visited, stack = set(), [start]
        while stack and R > 0:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for neighbor in graph_transpose.neighbors(node):
                    if R <= 0:
                        break
                    pp = model.prop_prob(node, neighbor, 1, use_attempts=False)
                    if random.random() <= pp:
                        stack.append(neighbor)
                    R -= 1
        return visited.union({start}), R


    def build_hypergraph(r_steps):
        if nx.is_directed(model.graph):
            transpose  = model.graph.reverse()
        else:
            transpose = model.graph.copy()
        h_graph = hypergraph()
        h_graph.add_nodes(transpose.nodes())
        
        while r_steps > 0:
            node_u = random.choice(list(transpose.nodes()))
            visited, r_steps = BIC_DFS(model, transpose, node_u, r_steps)
            h_graph.add_edge(visited)

        return h_graph
                    
    def build_seed_set(h_graph, n_seeds):
        seed_set = set()
        for i in range(n_seeds):
            degree_rank = {
                node_id: h_graph.degree(node_id)
                for node_id in h_graph.nodes
            }
            node_id = max(degree_rank, key=degree_rank.get)
            seed_set.add(node_id)
            h_graph.delete_node_and_incident_edges(node_id)
        return seed_set

    # NOTE: Rounds is the number of DFS steps allowed for simulation. From the paper:
    #       "This simulation process is performed as described in Section 2: we begin at a random node \(u\) and proceed 
    #        via  depth-first  search,  where  each  encountered  edge \(e\) is traversed independently with probability 
    #        \(p_e\)...  The  BuildHypergraph  subroutine  takes as input a bound R on its runtime; we continue building 
    #        edges  until  a  total  of \(R\) steps  has  been taken by the simulation process. (Note that the number of 
    #        steps taken  by the  process is equal to the number of edges considered by the depth-first search process). 
    #        Once  \(R\) steps  have  been  taken  in  total  over all simulations, we return the resulting hypergraph."
    model.prepare()
    m, n = model.graph.number_of_edges(), model.graph.number_of_nodes()
    rounds  = int(144 * (m + n) * epsilon ** (-3) * math.log(n, 2)) 
    h_graph = build_hypergraph(rounds)
    model.prepared = False
    return build_seed_set(h_graph, n_seeds)

