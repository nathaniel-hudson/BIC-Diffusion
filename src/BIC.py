import Algorithms
import Constants
import datetime
import networkx as nx
import pandas   as pd
import random   as rd

from scipy.stats import arcsine, uniform

"""
This file implements the Behavioral Independent Cascade (BIC) model for information
diffusion proposed by Hudson et al. 

This is implemented using the Python wrapper for the graph-tool API written in C++. 
NetworkX was shown to be too slow to be reliable for large network simulations.
"""

def append_dict(d1, d2):    
    keys = set(d1.keys()).union(set(d2.keys()))
    for key in keys:
        if key in d1:
            d1[key] += (d2[key])
        else:
            d1[key] = d2[key].copy()


def opinion_influence(graph, src, tar, opinion, t):
    return opinion[tar][t] * (1 - abs(opinion[src][0] - opinion[tar][t]))


def behavioral_influence(graph, node, ffm, coeffs=Constants.LAMBDA_DEFAULT):
    lambda_pos = {factor: coeffs[factor]
                  for factor in coeffs if coeffs[factor] >= 0}
    lambda_neg = {factor: coeffs[factor]
                  for factor in coeffs if coeffs[factor] < 0}

    beta_pos = 0
    beta_neg = 0

    for factor in lambda_pos: 
        beta_pos += lambda_pos[factor] * ffm[node][factor]
    for factor in lambda_neg:
        beta_neg += lambda_neg[factor] * ffm[node][factor]

    beta = beta_pos + beta_neg
    
    old_min = sum(lambda_neg.values()); new_min = 0.0
    old_max = sum(lambda_pos.values()); new_max = 1.0

    if old_min != old_max and new_min != new_max:
        influence = (((beta - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
    else:
        influence = (new_max + new_min) / 2
    
    return influence


def prop_prob(graph, source, target, attempt, opinion, ffm, t):
    if nx.is_directed(graph):
        neighbors = graph.predecessors(target)
    else:
        neighbors = graph.neighbors(target)

    opinion_inf = opinion_influence(graph, source, target, opinion, t)
    behavio_inf = behavioral_influence(graph, target, ffm)

    num = behavio_inf * opinion_inf
    den = sum([opinion_influence(graph, neighbor, target, opinion, t)
               for neighbor in neighbors])

    return 0 if den == 0 else (num/den) ** attempt


def general_update(graph, node, opinion, t):
    '''
    num = opinion[node][t]
    den = 1

    for neighbor in list(graph.iterInNeighbors(node)):
        # num += (1 - abs(opinion[node][t] - opinion[neighbor][t])) * opinion[neighbor][t]
        # den +=  1 - abs(opinion[node][t] - opinion[neighbor][t])
        num += opinion_influence(graph, neighbor, node, opinion, t) #* opinion[neighbor][t]
        den += opinion_influence(graph, neighbor, node, opinion, t) / opinion[node][t]

    return num/den
    '''
    if nx.is_directed(graph):
        neighbors = graph.predecessors(node)
    else:
        neighbors = graph.neighbors(node)

    num = opinion[node][0]
    den = 1
    for nei in neighbors:
        num += opinion[nei][t] * (1 - abs(opinion[node][0] - opinion[nei][t]))
        den += (1 - abs(opinion[node][0] - opinion[nei][t]))

    return num/den



def penalized_update(graph, node, opinion, t):
    '''
    num = opinion[node][t]
    den = 1

    for neighbor in list(graph.iterInNeighbors(node)):
        if opinion[neighbor][t] > opinion[node][t]:
            continue
        # num += (1 - abs(opinion[node][t] - opinion[neighbor][t])) * opinion[neighbor][t]
        # den +=  1 - abs(opinion[node][t] - opinion[neighbor][t])
        num += opinion_influence(graph, neighbor, node, opinion, t) #* opinion[neighbor][t]
        den += opinion_influence(graph, neighbor, node, opinion, t) / opinion[node][t]

    return num/den
    '''
    if nx.is_directed(graph):
        neighbors = graph.predecessors(node)
    else:
        neighbors = graph.neighbors(node)

    num = opinion[node][0]
    den = 1
    for nei in neighbors:
        if opinion[nei][t] > opinion[node][t]:
            continue
        num += opinion[nei][t] * (1 - abs(opinion[node][0] - opinion[nei][t]))
        den += (1 - abs(opinion[node][0] - opinion[nei][t]))

    return num/den


def diffuse(graph, active, attempts, opinion, ffm, t, penalized, killed, threshold=1):
    _active = set(active.copy())
    to_penalize = set()

    for node in set(active) - killed:
        neighbors = set(graph.neighbors(node))
        attempts[node] += 1
        for out_neighbor in neighbors - _active:
            if attempts[node] <= threshold:
                pp = prop_prob(graph, node, out_neighbor, attempts[node], opinion, ffm, t)
                if rd.random() <= pp:
                    _active.add(out_neighbor)
                else:#if penalized[out_neighbor] ==  False:
                    # penalized[out_neighbor] = True
                    to_penalize.add(out_neighbor)

            else:
                killed.add(node)

    for node in graph.nodes():
        if node in _active:
            opinion[node][t+1] = 1.0
        elif node in to_penalize:# or penalized[node] == True: # NOTE: Change this possibly?
            opinion[node][t+1] = penalized_update(graph, node, opinion, t)
        else:
            opinion[node][t+1] = general_update(graph, node, opinion, t)

    return _active


def add_record(data, record):
    data['trial'].append(record[0])
    data['time-step'].append(record[1])
    data['activated'].append(record[2])
    data['opinion'].append(record[3])
    data['algorithm'].append(record[4])


def simulate(graph, seeds, opinion, ffm, t_horizon, n_trials, algorithm='N/A', thresh=1):
    """[summary]
    
    Arguments:
        graph {[type]} -- [description]
        seeds {[type]} -- [description]
        opinion {[type]} -- [description]
        ffm {[type]} -- [description]
        t_horizon {[type]} -- [description]
        n_trials {[type]} -- [description]
    
    Keyword Arguments:
        algorithm {str} -- [description] (default: {'N/A'})
        thresh {int} -- [description] (default: {1})
    
    Returns:
        [type] -- [description]
    """
    NODES = graph.nodes()
    EDGES = graph.edges()

    total_opinion = lambda data, t: sum([data[node][t] for node in graph.nodes()])
    behavior = {node: behavioral_influence(graph, node, ffm) for node in graph.nodes()}

    data = {'trial': [], 'time-step': [], 'activated': [], 'opinion': [], 'algorithm': []}
    for trial in range(1, n_trials+1):
        active = seeds.copy()
        attempts = {node: 0 for node in NODES}
        killed = set()

        opinion_t = opinion.copy()
        penalized = {node: False for node in graph.nodes()}

        # Add the data for the 0th time-step, before diffusion begins.
        record = [trial, 0, 0, total_opinion(opinion_t, 0), algorithm]
        add_record(data, record)

        # Activate the seed nodes and polarize their opinion. Then record the data.
        for node in graph.nodes():
            opinion_t[node][1] = 1.0 if node in active else opinion_t[node][0]
        record = [trial, 1, len(active), total_opinion(opinion_t, 1), algorithm]
        add_record(data, record)

        # Iterate through each diffusion step, until the time horizon is exhausted.
        for t in range(1, t_horizon):
            # Run a *single* time-step of diffusion.
            active = diffuse(graph, active, attempts, opinion_t,
                             ffm, t, penalized, killed, thresh)

            # Record data from the diffusion step in `data`.
            record = [trial, t+1, len(active), total_opinion(opinion_t, t+1), algorithm]
            add_record(data, record)

    # Return a dict for the simulation.
    return data

# ============================================== #
# > WORKING DEMONSTRATION:                       #
#     Dummy example of using this file to run a  #
#     simple experiment simulation.             #
# ============================================== #
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import progressbar
    import seaborn as sns
    import time
    import warnings

    sns.set_style('darkgrid')
    warnings.filterwarnings('ignore')

    # Simulation setup.
    n_iters = 10
    n_trials = 10
    t_horizon = 50
    n_nodes = 200
    m = 10

    p = 0.125 # 0.09375
    budget = 10
    
    start = datetime.datetime.now()

    graph = nx.watts_strogatz_graph(10000, 60, 0.2)
    opinion = {node: {0: rd.random()} for node in graph.nodes()}
    ffm = {node: {'O': rd.random(), 'C': rd.random(), 'E': rd.random(), 
    'A': rd.random(), 'N': rd.random()} for node in graph.nodes()}
    seeds = [18]
    simulate(graph, seeds, opinion, ffm, 100, 1)
    
    end = datetime.datetime.now()

    print('Time elapsed:', end-start)
    exit(0)


    data = {}
    status = progressbar.ProgressBar(n_iters).start()

    start = time.time()
    for i in range(n_iters):
        status.update(i)
        graph = nx.barabasi_albert_graph(n_nodes, m)
        # graph = nx.erdos_renyi_graph(n_nodes, p, directed=True)
        # graph = nx.watts_strogatz_graph(n_nodes, m, p)

        # rand_gen = arcsine.rvs
        rand_gen = uniform.rvs
        # rand_gen = lambda: 0.0

        opinion = {node: {0: rand_gen()} for node in graph.nodes()}
        ffm = {node: {'O': rd.random(), 'C': rd.random(), 'E': rd.random(), 
        'A': rd.random(), 'N': rd.random()} for node in graph.nodes()}
        
        # Acquire seed nodes.
        dgr_seeds = Algorithms.degree(graph, budget)
        dis_seeds = Algorithms.degree_discount(graph, budget)
        iri_seeds = Algorithms.IRIE(graph, budget, opinion, ffm)
        min_seeds = Algorithms.min_opinion(graph, budget, opinion)
        opd_seeds = Algorithms.opinion_degree(graph, budget, opinion, ffm)
        opa_seeds = Algorithms.opinion_path(graph, budget, opinion, ffm)
        ran_seeds = Algorithms.random_sol(graph, budget)

        # Run the simulations.
        append_dict(data, simulate(graph, dgr_seeds, opinion, ffm, t_horizon, n_trials, 'Degree'))
        append_dict(data, simulate(graph, dis_seeds, opinion, ffm, t_horizon, n_trials, 'DegreeDisc'))
        append_dict(data, simulate(graph, iri_seeds, opinion, ffm, t_horizon, n_trials, 'IRIE'))
        append_dict(data, simulate(graph, min_seeds, opinion, ffm, t_horizon, n_trials, 'Min-Opinion'))
        append_dict(data, simulate(graph, opd_seeds, opinion, ffm, t_horizon, n_trials, 'Opinion-Deg'))
        append_dict(data, simulate(graph, opa_seeds, opinion, ffm, t_horizon, n_trials, 'Opinion-Path'))
        append_dict(data, simulate(graph, ran_seeds, opinion, ffm, t_horizon, n_trials, 'Random'))
    end = time.time()

    status.update(n_iters)

    df = pd.DataFrame.from_dict(data)

    sns.lineplot(y='opinion', x='time-step', data=df, hue='algorithm', marker='.')
    plt.title('Opinion Evolution (NX) -- %f sec.' % (end-start))
    plt.show()

    sns.lineplot(y='activated', x='time-step', data=df, hue='algorithm', marker='.')
    plt.title('Activation Evolution (NX) -- %f sec.' % (end-start))
    plt.show()
