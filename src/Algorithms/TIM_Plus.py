import networkx as nx
import math
import random
import time

from tqdm import tqdm

n_nodes = None
n_edges = None
RR_set = None
RR_deg = None # {node: [RR_set_ids...] for node in graph.nodes()}
theta = None
ell = None

def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def generate_random_RR(model, idx=None):
    directed = model.graph.is_directed()
    start_node = random.choice(list(model.graph.nodes()))
    queue = [start_node]
    visited = set(queue)

    if idx is not None:
        global RR_deg
        RR_deg[start_node].append(idx)

    while queue:
        v = queue.pop(0)
        if directed:
            neighbors = model.graph.predecessors(v)
        else:
            neighbors = model.graph.neighbors(v)

        for u in neighbors:
            visited.add(u)
            if idx is not None:
                RR_deg[u].append(idx)
            if random.random() < model.prop_prob(u, v, use_attempts=False):
                queue.append(u)

    return visited


def generate_random_RR_sets(model, num):
    global RR_deg
    RR_set = {}
    RR_deg = {node: [] for node in model.graph.nodes()}
    for idx in range(num):
        RR_set[idx] = generate_random_RR(model, idx)
    return RR_set


def identify_node_that_covers_most_RR_sets():
    return max(RR_set, key=RR_set.get)


def __alg1(model, n_seeds, theta):
    RR_set = generate_random_RR_sets(model, num=theta)
    seed_set = set()
    for j in range(n_seeds):
        node_v = identify_node_that_covers_most_RR_sets()
        seed_set.add(node_v)
        for RR in RR_deg[node_v]:
            if RR in RR_set:
                del RR_set[RR]
    return seed_set


def __alg2(model, n_seeds):
    global RR_set
    kpt_star = 1
    for i in range(int(math.log(n_nodes - 1, 2))):
        c_i = (6 * ell * math.log(n_nodes) + 6 * math.log(math.log(n_nodes, 2))) * 2 ** i
        c_i = int(c_i)

        RR_set = generate_random_RR_sets(model, num=c_i)
        directed = nx.is_directed(model.graph)
        degr_func = model.graph.in_degree if nx.is_directed(model.graph) \
            else model.graph.degree

        total = 0
        for RR in RR_set.values():
            weight = sum(degr_func(node) for node in RR)
            total += 1 - (1 - weight / n_edges) ** n_seeds

        for j in range(c_i):
            RR = generate_random_RR(model)

        if total/c_i > (0.5) ** i:
            kpt_star = n_nodes * total / (2 * c_i)
            return kpt_star
    return kpt_star


def __alg3(model, n_seeds, kpt_star, epsilon):
    global theta

    seed_set = set()
    for j in range(n_seeds):
        node_v = identify_node_that_covers_most_RR_sets()
        seed_set.add(node_v)
        for RR in RR_deg[node_v]:
            # RR_set.remove(RR)
            if RR in RR_set:
                del RR_set[RR]

    lmbda = (2 + epsilon) * ell * n_nodes * math.log(n_nodes) * epsilon ** -2
    theta = int(lmbda / kpt_star)
    new_RR_set = generate_random_RR_sets(model, num=theta)
    # f = fraction_of_RR_sets_covered_by_seed_set(new_RR_set, seed_set)
    covered_RR_sets = []
    for seed in seed_set:
        covered_RR_sets.extend(RR_deg[seed])
    f = len(set(covered_RR_sets)) / len(RR_set)
    kpt = f * n_nodes / (1 + epsilon)
    return max(kpt, kpt_star)


def TIM_plus_solution(model, n_seeds, epsilon=0.2, ELL=1):
    # Set global variables.
    global n_nodes, n_edges, ell
    n_nodes = model.graph.number_of_nodes()
    n_edges = model.graph.number_of_edges()
    ell = ELL

    # TIM+ Algorithm with the equations lifted from Tang et al.'s 2014 paper.
    kpt_star = __alg2(model, n_seeds)
    epsilon_prime = 5 * (ell * epsilon ** 2 / (n_seeds + ell)) ** (1/3)
    kpt_plus = __alg3(model, n_seeds, kpt_star, epsilon_prime)
    lmbda = (8 + 2 * epsilon) * n_nodes * (ell * math.log(n_nodes) + \
        math.log(nCr(n_nodes, n_seeds)) + math.log(2) * epsilon ** (-2))
    theta = int(lmbda / kpt_plus)
    seed_set = __alg1(model, n_seeds, theta)

    return seed_set

