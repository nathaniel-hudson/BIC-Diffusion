import networkx as nx


def find_next_pair(model, seed_set, max_iter, theta):
    is_seed = [False for node in model.graph.nodes()]
    for seed in seed_set:
        is_seed[seed] = True

    p_arr = [[0 for j in range(max_iter + 1)] for i in model.graph.nodes()]
    for i in model.graph.nodes():
        p_arr[i][1] = 1

    count = 0
    iteration = 2
    while (iteration < max_iter) and (count < len(model.graph)):
        # Number of nodes whose potential is updated in each iteration.
        count = 0

        for u in model.graph.nodes():
            if is_seed[u]:
                count += 1
                continue

            p = p_arr[u]
            k_out = nx.degree(model.graph, u)
            for v in model.graph.neighbors(u):
                w1 = 1.0 / k_out
                w2 = model.prop_prob(u, v, 1, use_attempts=False)
                potential1 = p_arr[v][iteration - 1]
                potential2 = p[iteration - 2]
                if (potential1 > theta) and (potential1 - w1 * potential2 > theta):
                    p[iteration] = p[iteration] + w2 * \
                        (potential1 - w1 * potential2)

            p[max_iter] = p[max_iter] + p[iteration]
            if p[iteration] <= theta:
                count += 1

        iteration += 1

    max_val = 0
    new_seed = -1
    for u in model.graph.nodes():
        if p_arr[u][max_iter] > max_val:
            max_val = p_arr[u][max_iter]
            new_seed = u

    return {"key": new_seed, "value": max_val}


def LAIM_solution(model, n_seeds, max_iter=5, theta=0.0001):
    seed_arr = [0 for i in range(n_seeds)]
    seed_set = set()
    total_score = 0

    for i in range(n_seeds):
        best_pair = find_next_pair(model, seed_set, max_iter+1, theta)
        seed_set.add(best_pair["key"])
        total_score += best_pair["value"]

    return seed_set


def fast_LAIM_solution(model, n_seeds, max_iter=2, theta=0.0001):

    max_iter += 1
    p_arr = [[0 for j in range(max_iter + 1)] for i in model.graph.nodes()]
    for i in model.graph.nodes():
        p_arr[i][1] = 1

    iteration = 2
    while (iteration < max_iter):
        for u in model.graph.nodes():
            p = p_arr[u]
            k_out = nx.degree(model.graph, u)
            for v in model.graph.neighbors(u):
                w1 = 1.0 / k_out
                w2 = model.prop_prob(u, v, 1, use_attempts=False)
                potential1 = p_arr[v][iteration - 1]
                potential2 = p[iteration - 2]
                if (potential1 > theta) and (potential1 - w1 * potential2 > theta):
                    p[iteration] = p[iteration] + w2 * \
                        (potential1 - w1 * potential2)
            p[max_iter] = p[max_iter] + p[iteration]
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
