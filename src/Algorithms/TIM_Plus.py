import networkx as nx
import random
import time

from math import factorial, log
from tqdm import tqdm

class Args:
    """This class is simply used to "package" the arguments that are commonly exchanged
       among the main three sub-procedures (node_selection, KPT_estimation, refine_KPT) to
       simplify the code. The main two variables that need some clarification are `RR_set`
       and `RR_deg`. The former, `RR_set`, will be a dictionary wherein keys are ids that
       map to a RR (reverse reachable) set. This is a dictionary to make indexing via ids
       simpler. The latter, `RR_deg`, is more important. This variable will be a dictionary
       wherein each key is a node id and the values are a set containing the RR ids the
       respective node was reached with.
    """
    def __init__(self, model, ell):
        self.model = model
        self.n_nodes = model.graph.number_of_nodes()
        self.n_edges = model.graph.number_of_edges()
        self.RR_set = None
        self.RR_deg = None 
        self.theta = None
        self.ell = ell

def nCr(n, r):
    return factorial(n) // factorial(r) // factorial(n - r)

def generate_random_RR(args, idx):
    directed = args.model.graph.is_directed()
    start_node = random.choice(list(args.model.graph.nodes()))
    queue = [start_node]
    visited = set(queue)

    while queue:
        v = queue.pop(0)
        if directed:
            neighbors = args.model.graph.predecessors(v)
        else:
            neighbors = args.model.graph.neighbors(v)

        for u in neighbors:
            visited.add(u)
            if idx is not None:
                args.RR_deg[u].add(idx)
            if random.random() < args.model.prop_prob(u, v, use_attempts=False):
                queue.append(u)

    return visited


def generate_random_RR_sets(args, num):
    args.RR_set = {}
    args.RR_deg = {node: set() for node in args.model.graph.nodes()}
    for idx in range(num):
        args.RR_set[idx] = generate_random_RR(args, idx)


def identify_node_that_covers_most_RR_sets(args, covered_set):
    """Greedy solution to the maximum coverage problem.

    Parameters
    ----------
    args : Args
        Wrapper for the arguments used across the sub-procedures.
    covered_set : set
        Set of RR sets that have already been covered by previously seeded nodes.

    Returns
    -------
    int
        Greedily picked key for maximum coverage.
    """
    max_key = 0
    max_val = 0
    for key in args.RR_deg:
        val = len(args.RR_deg[key] - covered_set)
        if val > max_val:
            max_key = key
            max_val = val
    return max_key


def node_selection(args, n_seeds, theta):
    """Implementation of Algorithm 1.

    Parameters
    ----------
    args : Args
        Wrapper for the arguments used across the sub-procedures.
    n_seeds : int
        Number of seed nodes.
    theta : int
        Number of RR sets to be generated.
        
    Returns
    -------
    set
        Seed node set.
    """
    generate_random_RR_sets(args, num=theta)
    seed_set = set()
    covered_set = set()
    for k in range(n_seeds):
        node_v = identify_node_that_covers_most_RR_sets(args, covered_set)
        seed_set.add(node_v)
        for RR in args.RR_deg[node_v]:
            covered_set.add(RR)
            if RR in args.RR_set:
                del args.RR_set[RR]
                ## TODO: We need to remove the RR values from each node's RR_deg set.
            
        # Avoid redundant/duplicate seed choices.
        args.RR_deg[node_v].clear()
            
    return seed_set


def KPT_estimation(args, n_seeds):
    """Implementation of Algorithm 2.

    Parameters
    ----------
    args : Args
        Wrapper for the arguments used across the sub-procedures.
    n_seeds : int
        Number of seed nodes.
        
    Returns
    -------
    float
        KPT* value which is the later to calculate number of needed RR sets.
    """
    directed = nx.is_directed(args.model.graph)
    degr_func = args.model.graph.in_degree \
                if nx.is_directed(args.model.graph) \
                else args.model.graph.degree
    kpt_star = 1
    for i in range(int(log(args.n_nodes - 1, 2))):
        c_i = (6 * args.ell * log(args.n_nodes) + 6 * log(log(args.n_nodes, 2))) * 2 ** i
        c_i = int(c_i)  
        total_sum = 0

        generate_random_RR_sets(args, num=c_i)
        for R in args.RR_set.values():
            weight = sum(degr_func(node) for node in R)
            total_sum += 1 - (1 - weight / args.n_edges) ** n_seeds

        if total_sum / c_i > (0.5) ** i:
            kpt_star = args.n_nodes * total_sum / (2 * c_i)
            return kpt_star

    return kpt_star


def refine_KPT(args, n_seeds, kpt_star, epsilon_prime):
    """Implementation of Algorithm 3.
    
    Parameters
    ----------
    args : Args
        Wrapper for the arguments used across the sub-procedures.
    n_seeds : int
        Number of seed nodes.
    kpt_star : float
        KPT* parameter computed from KPT_estimation (or Algorithm 2).
    epsilon_prime : float
        Parameter used to minimize the total number of RR sets for this sub-procedure 
        and node_selection.
        
    Returns
    -------
    float
        KPT+ value which is the maximized KPT value upon this refinement sub-procedure.
    """
    seed_set = set()
    covered_set = set()
    for j in range(n_seeds):
        node_v = identify_node_that_covers_most_RR_sets(args, covered_set)
        seed_set.add(node_v)
        # Remove RR sets covered by node_v.
        for RR in args.RR_deg[node_v]:
            covered_set.add(RR)
            if RR in args.RR_set:
                del args.RR_set[RR]
        # Avoid redundant/duplicate seed choices.
        args.RR_deg[node_v].clear()

    lambda_prime = (2 + epsilon_prime) * args.ell * args.n_nodes * \
        log(args.n_nodes) * epsilon_prime ** -2
    theta_prime = int(lambda_prime / kpt_star)
    generate_random_RR_sets(args, num=theta_prime)
    
    covered_RR_sets = []
    for seed in seed_set:
        covered_RR_sets.extend(args.RR_deg[seed])
    f = len(set(covered_RR_sets)) / len(args.RR_set)
    kpt_prime = f * args.n_nodes / (1 + epsilon_prime)
    return max(kpt_prime, kpt_star)


def TIM_plus_solution(model, n_seeds, epsilon=0.2, ell=1):
    # This wrapper is used to simplify the exchange of shared data-structures across the 
    # sub-procedures (largely the `model` and the RR sets).
    args = Args(model, ell)

    # TIM+ Algorithm with the equations lifted from Tang et al.'s 2014 paper.
    kpt_star = KPT_estimation(args, n_seeds)
    epsilon_prime = 5 * (args.ell * epsilon ** 2 / (n_seeds + args.ell)) ** (1/3)
    kpt_plus = refine_KPT(args, n_seeds, kpt_star, epsilon_prime)
    lmbda = (8 + 2 * epsilon) * args.n_nodes * (args.ell * log(args.n_nodes) + \
        log(nCr(args.n_nodes, n_seeds)) + log(2) * epsilon ** (-2))
    theta = int(lmbda / kpt_plus)
    seed_set = node_selection(args, n_seeds, theta)

    return seed_set