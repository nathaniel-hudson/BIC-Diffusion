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
            dist[u][v] = model.prop_prob(u, v, use_attempts=False)
    else:
        for (u,v) in model.graph.edges():
            dist[u][v] = model.prop_prob(u, v, use_attempts=False)
            dist[u][v] = model.prop_prob(v, u, use_attempts=False)
    
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



def new_solution(model, n_seeds, max_iter=2, theta=0.0001):
    
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
                w1 = 1.0 / k_out
                pp_vu = model.prop_prob(v, u, use_attempts=False)
                pp_uv = model.prop_prob(u, v, use_attempts=False)
                cand1 = p_arr[v][i - 1]
                cand2 = p_arr[u][i - 2]
                # if (cand1 > theta) and (cand1 - w1 * cand2 > theta):
                #     p_arr[u][i] = p_arr[u][i] + pp_uv * (cand1 - w1 * cand2) + \
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
                w1 = 1.0 / k_out
                w2 = model.prop_prob(u, v, use_attempts=False)
                potential1 = p_arr[v][iteration - 1]
                potential2 = p_arr[u][iteration - 2]
                if (potential1 > theta) and (potential1 - w1 * potential2 > theta):
                    p_arr[u][iteration] = p_arr[u][iteration] + w2 * \
                        (potential1 - w1 * potential2)
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
            p_arr[node][1] = 1
        else:
            p_arr[node][1] = 0 # NOTE: We may want to make this negative? Just an idea.

    iteration = 2
    while (iteration < max_iter):
        for u in model.graph.nodes():
            k_out = nx.degree(model.graph, u)
            for v in model.graph.neighbors(u):
                w1 = 1.0 / k_out
                w2 = model.prop_prob(u, v, use_attempts=False)
                potential1 = p_arr[v][iteration - 1]
                potential2 = p_arr[u][iteration - 2]
                if (potential1 > theta) and (potential1 - w1 * potential2 > theta):
                    p_arr[u][iteration] = p_arr[u][iteration] + w2 * \
                        (potential1 - w1 * potential2)
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


def nom_solution(model, n_seeds, psi=0.4):
    pruned_nodes = set([u for u in model.graph.nodes() if model.init_opinion[u] <= psi])
    # pn_set = set(pruned_nodes) ## We do this to simplify lookup for below.
    # pruned_edges = [(u,v) for (u,v) in model.graph.edges() if (u in pn_set) and (v in pn_set)]
    # mapping_from = {idx: pruned_nodes[idx] for idx in range(len(pruned_nodes))}
    # mapping_to   = {pruned_nodes[idx]: idx for idx in range(len(pruned_nodes))}

    # graph_psi = nx.DiGraph() if model.graph.is_directed() else nx.Graph()
    # graph_psi.add_nodes_from(pruned_nodes)
    # graph_psi.add_edges_from(pruned_edges)

    # graph_psi = nx.relabel_nodes(graph_psi, mapping_to)
    # opinion = [model.init_opinion[node] for node in pruned_nodes]
    # ffm = {node: model.ffm[node] for node in pruned_nodes}
    # model_psi = BIC(graph_psi, ffm, opinion)

    return __fast_LAIM(model, n_seeds, pruned_nodes)
