import Algorithms
import datetime
import ego_io
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import os.path
import pandas as pd
import progressbar
import random as rd
import seaborn as sns
import time
import warnings 

from BIC import *
from scipy.stats import arcsine, uniform

warnings.filterwarnings('ignore')
sns.set_style('darkgrid')


def append_dict(d1, d2):
    keys = set(d1.keys()).union(set(d2.keys()))
    for key in keys:
        if key in d1.keys():
            d1[key] += (d2[key])
        else:
            d1[key] = d2[key].copy()


def init_opinion(graph, communities):
    community_opinion = {i: rd.random() for i in communities}
    node_opinion = {node: {0: []} for node in graph.nodes()}

    for i in communities:
        for node in communities[i]:
            node_opinion[node][0].append(community_opinion[i])

    for i in node_opinion:
        node_opinion[i][0] = np.mean(node_opinion[i][0])

    return node_opinion


def init_ffm(graph):
    ffm = {node: {'O': rd.random(), 'C': rd.random(), 'E': rd.random(),
                  'A': rd.random(), 'N': rd.random()} for node in graph.Nodes()}
    return ffm


def init_graph(label):
    # STEP 1: Open the graph.
    g_type = nx.DiGraph if TOPOS[label]['dir'] else nx.Graph
    graph  = nx.read_edgelist(TOPOS[label]['topo'], nodetype=int, create_using=g_type)

    # STEP 2: Read in the communities.
    if label == 'amazon' or label == 'dblp':
        communities = {}
        with open(TOPOS[label]['comm'], 'r') as f:
            i = 0
            line = f.readline()
            while line:
                communities[i] = [int(node) for node in line.split()]
                line = f.readline()
                i += 1

    elif label == 'eu-core':
        communities = {}
        with open(TOPOS[label]['comm'], 'r') as f:
            line = f.readline()
            while line:
                temp = line.split()
                node, community = int(temp[0]), int(temp[1])
                if community not in communities:
                    communities[community] = [node]
                else:
                    communities[community].append(node)
                line = f.readline()

    # STEP 3: Initialize the nodewise opinion and FFM features w.r.t. community.
    opinion = init_opinion(graph, communities)
    ffm = init_ffm(graph)

    return graph, opinion, ffm


def run_experiment(graph, opinion, ffm, seed_set, t_horizon, n_trials):
    data = {}
    for seeds, alg_name in seed_set:
        sim_data = simulate(graph, seeds, opinion, ffm, t_horizon, n_trials, alg_name)
        append_dict(data, sim_data)
    return data


def rand_graph():
    graph = nx.erdos_renyi_graph(50, 0.125, directed=True)
    opinion = {node: {0: rd.random()} for node in graph.nodes()}
    ffm = init_ffm(graph)
    return graph, opinion, ffm


def get_seed_set(graph, budget, opinion, ffm):
    degree_seeds = Algorithms.degree(graph, budget)
    degdis_seeds = Algorithms.degree_discount(graph, budget)
    irie_seeds   = Algorithms.IRIE(graph, budget, opinion, ffm)
    op_deg_seeds = Algorithms.opinion_degree(graph, budget, opinion, ffm)
    random_seeds = Algorithms.random_sol(graph, budget)
    seed_set = [(degree_seeds, 'Degree'), (degdis_seeds, 'DegreeDiscount'), (irie_seeds, 'IRIE'), 
    (op_deg_seeds, 'OpinionDegree'), (random_seeds, 'Random')]
    return seed_set


def range_experiment(n_nodes, n_edges, seed_range, n_iters, n_trials, t_horizon, subdir_prefix='', WS_p=0.125):
    graph_gens = ['BA', 'ER', 'WS']

    if subdir_prefix != '':
        subdir_prefix += '__'

    subdir = subdir_prefix + str(datetime.datetime.now()).replace(':', ';')
    path = os.path.join('out', 'data', 'synthetic', subdir)
    if not os.path.exists(path):
        os.mkdir(path)
        print('Made path: %s' % str(path))

    maxval = len(seed_range) * len(graph_gens) * n_iters * 2 * 2
    status = progressbar.ProgressBar(maxval)
    status.start()
    status_val = 0

    for graph_type in graph_gens:
        print('>>> Loading graph (%s) and communities.' % graph_type)
        if graph_type == 'BA':
            graph = nx.barabasi_albert_graph(n_nodes, n_edges//n_nodes)
        elif graph_type == 'ER':
            graph = nx.gnm_random_graph(n_nodes, n_edges)
        elif graph_type == 'WS':
            m = n_edges // n_nodes
            if m % 2 != 0:
                m += 1
            graph = nx.watts_strogatz_graph(n_nodes, m, WS_p)

        circles = nx.algorithms.community.greedy_modularity_communities(graph)
        circles = {i: circles[i] for i in range(len(circles))}

        if nx.is_directed(graph) == False:
            graph = nx.to_directed(graph)

        for k in seed_range:
            for rand_gen, rand_label in [(arcsine.rvs, 'polarized'), (uniform.rvs, 'uniform')]:
                for homophilic in [False, True]:
                    data = {}
                    for i in range(n_iters):
                        # Initialize opinion and FFM parameters.
                        print('>>> Initializing opinion and FFM parameters (homophily=%s, mood=%s).' % (str(homophilic), rand_label))
                        if homophilic == True: opinion = ego_io.init_opinion(graph, circles, rand_gen=rand_gen)
                        else:                  opinion = ego_io.init_opinion(graph, rand_gen=rand_gen)
                        ffm = ego_io.init_ffm(graph)

                        # Generate seed sets and then run the simulation, storing the results.
                        print('>>> Generating seed sets, with k=%d.' % k)
                        seed_set = get_seed_set(graph, k, opinion, ffm)
                        print('>>> Running simulation.')
                        append_dict(data, run_experiment(graph, opinion, ffm, seed_set, t_horizon, n_trials))
                        print('>>> Simulation done!  (%0.2f%%)\n' % (100 * ((status_val+1)/maxval)))

                        # Update the status of the simulation.
                        status_val += 1 
                        status.update(status_val)

                    # Output the results for these iterations of the experiment.
                    filename = 'topo=%s_k=%d_homophilic=%s_%s.csv' % (graph_type, k, str(homophilic), rand_label)
                    data['topo'] = [graph_type for j in range(len(data['opinion']))]
                    data['k'] = [k for j in range(len(data['opinion']))]
                    dataframe = pd.DataFrame.from_dict(data)
                    path = os.path.join('out', 'data', 'synthetic', subdir, filename)
                    dataframe.to_csv(path)

    status.update(status.maxval)

    # Output a README document with information about the simulations for these experiments.
    readme = ['These simulations were run for a seed range of `%s` over `n_trials=%d` and `n_iters=%d`, with `t_horizon=%d`.' % (str(seed_range), n_trials, n_iters, t_horizon)]
    readme.append('These simulations considered the following network topologies: %s. With |V|=%d and |E|~=%d.' % (graph_gens, n_nodes, n_edges))
    readme.append('Additionally, these simulations considered the following algorithms with budget: %s.' % str([tup[1] for tup in seed_set]))
    with open(os.path.join('out', 'data', 'synthetic', subdir, 'README.txt'), 'w') as f:
        for line in readme:
            f.write(line + '\n')



N_NODES_DEFAULT   = 300
N_EDGES_DEFAULT   = 12*N_NODES_DEFAULT
K_MAX_DEFAULT     = 25
K_DELTA_DEFUALT   = 5
N_ITERS_DEFAULT   = 10
N_TRIALS_DEFAULT  = 10
T_HORIZON_DEFAULT = 50

if __name__ == '__main__':
    print('')
    ans = input('Use standard/default paramets? (y/n): ').lower()
    while ans != 'y' and ans != 'n':
        ans = input('ERR - Use standard/default paramets? (y/n): ').lower()
    
    if ans == 'y':
        n_nodes = N_NODES_DEFAULT
        n_edges = N_EDGES_DEFAULT
        k_max = K_MAX_DEFAULT
        k_delta = K_DELTA_DEFUALT
        n_iters = N_ITERS_DEFAULT
        n_trials = N_TRIALS_DEFAULT
        t_horizon = T_HORIZON_DEFAULT
    else:
        n_nodes   = int(input('n_nodes:   ')) # 500
        n_edges   = int(input('n_edges:   ')) # 10000
        k_max     = int(input('k_max:     ')) # 50
        k_delta   = int(input('k_delta:   ')) # 10
        n_iters   = int(input('n_iters:   ')) # 1#00
        n_trials  = int(input('n_trials:  ')) # 1#00
        t_horizon = int(input('t_horizon: ')) # 50

    start = time.time()
    range_experiment(n_nodes, n_edges, range(0, k_max+1, k_delta), n_iters, n_trials, t_horizon, 'TENTATIVE')
    end = time.time()
    print('Time elapsed: %s sec.' % (end-start))