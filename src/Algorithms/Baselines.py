import networkx as nx
import random

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


def TIM_solution(model, n_seeds):
    """Implementation of the TIM^{+} algorithm proposed by Borgs et al. in 2014 in the paper entitled "Maximizing Social Influence in Nearly Optimal Time".

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
    def _build_hypergraph(rounds):
        transpose  = model.graph.reverse()
        hypergraph = nx.Graph()
        hypergraph.add_nodes_from(model.graph)
        hyperedge = 0
        for _ in range(rounds):
            discovered_nodes = set()
            model.prepare()
            timestep = 1

            while True:
                u = random.choice(graph.model.nodes)
                _, visited = model.diffuse(set([u]), timestep)
                timestep += 1
                if visited == discovered_nodes:
                    break
                else:
                    discovered_nodes = discovered_nodes.union(visited)

                for v in discovered_nodes:
                    if u != v:
                        hypergraph.add_edge(u, f'hyperedge-{hyperedge}')
                hyperedge += 1
                    


                

    m, n = model.graph.number_of_edges, model.graph.number_of_nodes
    rounds = 144 * (m + n) * epsilon ** (-3) * math.log(n, base=2)
    hypergraph = build_hypergraph(R)
    return build_set_set(H, n_seeds)
    # return set()

