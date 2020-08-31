import networkx as nx
import numpy as np

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


def proto1_solution(model, n_seeds, theta=0.2):
    """First prototype... Completely failed. It was both terribly slow (slower than TIM+)
       and completely lacking in terms of providing good results. 

    Parameters
    ----------
    model : BIC
        [description]
    n_seeds : int
        Number of seeds
    theta : float, optional
        Activation path saturation point, by default 0.2

    Returns
    -------
    [type]
        [description]
    """

    model.prepare()
    exp_op = [
        [0 for v in model.graph.nodes()]
        for u in model.graph.nodes()
    ]

    ## NOTE: This part makes sense to me... Maybe a matrix structure instead?
    def gamma(u, ap=1.0):
        if ap <= theta:
            return 0

        value = 1.0 - model.init_opinion[u]
        for v in model.graph.neighbors(u):
            pp = model.prop_prob(u, v, use_attempts=False)
            temp = pp * gamma(v, ap * pp) + (1 - pp) * model.penalized_update(v, 0)
            # temp = pp * gamma(v, ap - 1) ## NOTE: Hop-based approach.
            exp_op[u][v] = temp
            value += temp

        return value

    # for node 
    for u in model.graph.nodes():
        exp_op[u][u] = gamma(u)

    seed_set = set()
    for seed in range(n_seeds):
        u = np.argmax([exp_op[i][i] for i in model.graph.nodes()])
        seed_set.add(u)
        ## TODO: Now we need to come up with a way to simply anneal the values
        ##       in `exp_op` based on the newly seeded node to avoid redundant
        ##       choices.
        for v in model.graph.neighbors(u):
            exp_op[v][v] -= exp_op[u][v]
        # del exp_op[u]

    model.prepared = False
    return seed_set



def new_solution_bad(model, n_seeds, max_iter=2, theta=0.0001):
    
    model.prepare()

    """ VERSION 1. EXTREMELY expensive computationally -- not worth it.
    def step(u, i):
        if i < 0:
            return 0
        elif i == 0:
            return 1 - model.init_opinion[u]
        elif i > 0:
            sum_total = 0
            for v in model.graph.neighbors(u):
                pp_uv = model.prop_prob(u, v, use_attempts=False)
                pp_vu = model.prop_prob(v, u, use_attempts=False)
                sum_total += pp_uv * (step(v, i-1) - pp_vu * step(u, i-2)) + \
                    (1-pp_uv) * ((model.penalized_update(v, 0) - model.init_opinion[v]) - \
                    (1-pp_vu) *  (model.penalized_update(u, 0) - model.init_opinion[u]))
            return sum_total

    scores = [step(node, max_depth) for node in model.graph.nodes()]
    seed_set = set()
    for _ in range(n_seeds):
        seed = np.argmax(scores)
        seed_set.add(seed)
        scores[seed] = float("-inf")
    model.prepared = False
    return seed_set
    """

    """ VERSION 2. Alright in all but one case: polarized, community-aware.
    if n_seeds < 1:
        return set()
        
    max_iter += 1
    p_arr = [[0 for j in range(max_iter + 1)] for i in model.graph.nodes()]
    for node in model.graph.nodes():
        p_arr[node][1] = 1.0 - model.init_opinion[node]

    i = 2
    while (i < max_iter):
        for u in model.graph.nodes():
            k_out = nx.degree(model.graph, u)
            for v in model.graph.neighbors(u):
                pp_vu = 1.0 / k_out
                pp_vu = model.prop_prob(v, u, use_attempts=False)
                pp_uv = model.prop_prob(u, v, use_attempts=False)
                cand1 = p_arr[v][i - 1]
                cand2 = p_arr[u][i - 2]
                # if (cand1 > theta) and (cand1 - pp_vu * cand2 > theta):
                #     p_arr[u][i] = p_arr[u][i] + pp_uv * (cand1 - pp_vu * cand2) + \
                if (cand1 > theta) and (cand1 - pp_vu * cand2 > theta):
                    adv = p_arr[u][i] + pp_uv * (cand1 - pp_vu * cand2)
                    pen = (1-pp_uv) * (model.penalized_update(v, 0) - model.init_opinion[v])
                    p_arr[u][i] =  adv + pen
                         # + \
                        # (1-pp_vu) * (model.penalized_update(u, 0) - model.init_opinion[u]))
                    
            p_arr[u][max_iter] = p_arr[u][max_iter] + p_arr[u][i]
        i += 1

    seeds = set()
    for i in range(n_seeds):
        max_val  = float("-inf")
        new_seed = None
        for j in model.graph.nodes():
            if (p_arr[j][max_iter] > max_val):
                max_val = p_arr[j][max_iter]
                new_seed = j
        seeds.add(new_seed)
        p_arr[new_seed][max_iter] = 0

    # print(f"seeds -> {seeds}")
    return seeds
    """


    """
    ## GREEDY SOLUTION
    seed_set = set()
    for k in range(n_seeds):
        values = {
            node: model.simulate(seed_set.union({node}), 100)[0] 
            for node in set(model.graph.nodes()) - seed_set
        }
        seed_set.add(max(values, key=values.get))
    return seed_set
    """

    if n_seeds < 1:
        return set()
        
    max_iter += 1
    p_arr = [[0 for j in range(max_iter + 1)] for i in model.graph.nodes()]
    for i in model.graph.nodes():
        p_arr[i][1] = 1

    iteration = 2
    while (iteration < max_iter):
        for u in model.graph.nodes():
            k_out = nx.degree(model.graph, u)
            for v in model.graph.neighbors(u):
                pp_vu = 1.0 / k_out
                pp_uv = model.prop_prob(u, v, use_attempts=False)
                cand1 = p_arr[v][iteration - 1]
                cand2 = p_arr[u][iteration - 2]
                if (cand1 > theta) and (cand1 - pp_vu * cand2 > theta):
                    p_arr[u][iteration] = p_arr[u][iteration] + pp_uv * \
                        (cand1 - pp_vu * cand2)
            p_arr[u][max_iter] = p_arr[u][max_iter] + p_arr[u][iteration]
        iteration += 1

    for i in model.graph.nodes():
        p_arr[i][1] *= (1 - model.init_opinion[i])

    seeds = set()
    for i in range(n_seeds):
        max_val = 0
        new_seed = -1
        for j in model.graph.nodes():
            if (p_arr[j][max_iter] > max_val):
                max_val = p_arr[j][max_iter]
                new_seed = j
        seeds.add(new_seed)
        p_arr[new_seed][max_iter] = 0

    return seeds


def __fast_LAIM(model, n_seeds, pruned_nodes, max_iter=2, theta=0.0001):
    if n_seeds < 1:
        return set()

    max_iter += 1
    p_arr = [[0 for j in range(max_iter + 1)] for i in model.graph.nodes()]
    for node in model.graph.nodes():
        if node in pruned_nodes:
            p_arr[node][1] = (1 - model.init_opinion[node]) ** 2 #1
        else:
            p_arr[node][1] = 0#-1 # NOTE: We may want to make this negative? Just an idea.

    iteration = 2
    while (iteration < max_iter):
        for u in model.graph.nodes():
            k_out = nx.degree(model.graph, u)
            for v in model.graph.neighbors(u):
                pp_uv = model.prop_prob(u, v, use_attempts=False)
                pp_vu = model.prop_prob(v, u, use_attempts=False) # 1.0 / k_out
                cand1 = p_arr[v][iteration - 1]
                cand2 = p_arr[u][iteration - 2]
                if (cand1 > theta) and (cand1 - pp_vu * cand2 > theta):
                    p_arr[u][iteration] = p_arr[u][iteration] + pp_uv * (cand1 - pp_vu * cand2)
            p_arr[u][max_iter] = p_arr[u][max_iter] + p_arr[u][iteration]
        iteration += 1

    seeds = set()
    total_score = 0
    for i in range(n_seeds):
        max_val = 0
        new_seed = -1
        for j in model.graph.nodes():
            if (p_arr[j][max_iter] > max_val):
                max_val = p_arr[j][max_iter]
                new_seed = j
        seeds.add(new_seed)
        total_score += max_val
        p_arr[new_seed][max_iter] = 0
        
    return seeds


def new_solution(model, n_seeds, psi=0.5):
    pruned_nodes = set([u for u in model.graph.nodes() if model.init_opinion[u] <= psi])
    seeds = __fast_LAIM(model, n_seeds, pruned_nodes)
    return seeds
