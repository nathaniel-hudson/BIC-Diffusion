import networkx as nx
import math
import random
import time

from tqdm import tqdm

# from .hypergraph import hypergraph

class ArgWrapper:

    def __init__(self, model, n_seeds, epsilon):
        ## Initialize necessary algorithm variables.
        self.hypergraph = [[] for node in model.graph.nodes()]
        self.hypergraph_t = []
        self.hyperID = 0
        self.rand_seed = None
        self.visit_mark = [0 for node in model.graph.nodes()]
        self.visit = [False for node in model.graph.nodes()]
        self.epsilon = epsilon

        self.graph_t = model.graph if (nx.is_directed(model.graph) == False) else model.graph.reverse()
        self.model = model
        self.n_edges = model.graph.number_of_edges()
        self.n_nodes = model.graph.number_of_nodes()
        self.n_seeds = n_seeds


def kpt_estimation(args):
    
    def mg_t(u):
        return 1.0 * build_hypergraph_node(args, u, 0, False)
    
    # algorithm2()
    lb, c = 0.5, 0
    x = 0 ## NOTE: Delete later
    while True:
        loop = int((6 * math.log(args.n_nodes) + 6 * math.log(math.log(args.n_nodes) / math.log(2))) * 1 / lb)
        c = 0
        sum_mg_tu = 0
        for i in range(loop):
            u = random.choice(list(args.model.graph))
            mg_tu = mg_t(u)
            pu = mg_tu / args.n_edges
            sum_mg_tu += mg_tu
            c += 1 - (1 - pu) ** args.n_seeds
        c /= loop
        print(f"TIM.kpt_estimation(): {c} > {lb} -> {c > lb} {x}"); x += 1
        if c > lb:
            break
        lb /= 2
    
    # kpt_estimation()
    ept = c * args.n_nodes
    return ept / 2


def build_seed_set(args):
    degree = []
    unvisited = [True for i in range(len(args.hypergraph_t))]
    seed_set = set()

    for i in range(args.n_nodes):
        degree.append(len(args.hypergraph[i]))

    for i in range(args.n_seeds):
        seed = degree.index(max(degree))
        seed_set.add(seed)
        degree[seed] = 0
        for t in args.hypergraph[seed]:
            if unvisited[t]:
                unvisited[t] = False
                for item in args.hypergraph_t[t]:
                    degree[item] -= 1

    return seed_set


def build_hypergraph_node(args, u_start, hyperiiid, add_hyper_edge):
    n_visit_edge = 1
    if add_hyper_edge:
        args.hypergraph_t[hyperiiid].append(u_start)

    n_visit_mark = 0
    queue = [u_start]
    args.visit_mark[n_visit_mark] = u_start
    n_visit_mark += 1
    args.visit[u_start] = True

    while len(queue) > 0:
        u = queue.pop(0)
        for v in nx.neighbors(args.graph_t, u):
            n_visit_edge += 1
            if random.random() > args.model.prop_prob(u, v, use_attempts=False):
                continue
            if args.visit[v]:
                continue
            else:
                args.visit_mark[n_visit_mark] = v; n_visit_mark += 1
                args.visit[v] = True

            queue.append(v)
            if add_hyper_edge:
                args.hypergraph_t[hyperiiid].append(v)

    for i in range(n_visit_mark):
        args.visit[args.visit_mark[i]] = False
    return n_visit_edge


def build_hypergraph(args, R): ## NOTE: EXTREMELY slow...
    args.hyperID = R
    args.hypergraph   = [[] for _ in args.model.graph.nodes()]
    args.hypergraph_t = [[] for _ in range(R)]

    # for idx in range(R):
    for idx in tqdm(range(R)):
        node_u = random.choice(list(args.model.graph))
        build_hypergraph_node(args, node_u, idx, True)

    total_added_element = 0
    for i in range(R):
        for t in args.hypergraph_t[i]:
            args.hypergraph[t].append(i)
            total_added_element += 1


def influence_hypergraph(args, seed_set):
    s = set()
    for t in seed_set:
        for tt in args.hypergraph[t]:
            s.add(tt)
    influence = args.n_nodes * len(s) / args.hyperID
    return influence


def logcnk(n_nodes, n_seeds):
    ans = 0
    for i in range(n_nodes-n_seeds+1, n_nodes+1):
        ans += math.log(i)
    for i in range(i, n_seeds+1):
        ans -= math.log(i)
    return ans


def refine_KPT(args, epsilon, ept):
    R = int((2 + epsilon) * (args.n_nodes * math.log(args.n_nodes)) / (epsilon ** 2 * ept))
    build_hypergraph(args, R)


def node_selection(args, epsilon, opt):
    R = int((8 + 2 * epsilon) * (math.log(args.n_nodes) + math.log(2) + \
            args.n_nodes * logcnk(args.n_nodes, args.n_seeds)) / (epsilon ** 2 * opt))
    build_hypergraph(args, R)
    return build_seed_set(args)


def TIM_solution(model, n_seeds, epsilon=0.2):
    ### NOTE: Use epsilon = 0.2 and ell = 1 (from the abstract of the paper)?
    # Initialize argument wrapper. This is to simplify the arg-passing  across the helper functions.
    args = ArgWrapper(model, n_seeds, epsilon)    

    # PART 1: KPT Estimation and then refiniing the KPT value.
    kpt_star = kpt_estimation(args)

    # PART 2: Build initial seed set and measure the influence and the KPT score.
    eps_prime = 5 * (args.epsilon ** 2 / (args.n_seeds+1)) ** (1/3)
    refine_KPT(args, eps_prime, kpt_star)
    seed_set = build_seed_set(args)
    kpt = influence_hypergraph(args, seed_set) 
    kpt /= 1 + eps_prime
    kpt_plus = max(kpt, kpt_star)

    # PART 3: Node selection.
    seed_set = node_selection(args, epsilon, kpt_plus)
    return seed_set