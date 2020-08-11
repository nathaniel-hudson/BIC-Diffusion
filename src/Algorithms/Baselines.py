import networkx as nx
import math
import random

from .hypergraph import hypergraph

# from halp.undirected_hypergraph import UndirectedHypergraph

def IMM_solution(model, n_seeds):
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

    def build_hypergraph(rounds):
        transpose  = model.graph.reverse()
        h_graph = hypergraph()
        h_graph.add_nodes(model.graph.nodes())
        
        for _ in range(rounds):
            seed = set(random.choice(model.graph.nodes()))
            model.prepare()
            _, _, visited = model.simulate(seed, time_horizon)
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
    #        edges  until  a  total  of \(R\) steps has been taken by the simula- tion process. (Note that the number of 
    #        steps taken  by the  process is equal to the number of edges considered by the depth-first search process). 
    #        Once  \(R\) steps  have  been  taken  in  total  over all simulations, we return the resulting hypergraph."
    m, n = model.graph.number_of_edges(), model.graph.number_of_nodes()
    rounds  = 144## int(144 * (m + n) * epsilon ** (-3) * math.log(n, 2)) 
    h_graph = build_hypergraph(rounds)
    return build_seed_set(h_graph, n_seeds)

