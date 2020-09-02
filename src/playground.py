import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import random
import seaborn as sns
import time
import Simulation

from scipy.stats           import arcsine
from Algorithms.LAIM       import *
from Algorithms.TIM_Plus   import TIM_plus_solution
from Algorithms.Proposed   import *
from Algorithms.Heuristics import *

from Diffusion.Model import BIC

"""
Using the new implementation of TIM+... On a BA network with 1000 nodes and K=32, we have the following runtimes and we
consider n_seeds=5:
    TIM+ runtime      746.75121
    LAIM runtime      57.85623
    FastLAIM runtime  2.84473
"""

def similarity(graph, opinion, f):
    ratios = []
    start = time.time()
    for node in graph.nodes():
        neighbors = graph.neighbors(node)
        opinions = [opinion[neighbor] for neighbor in neighbors]
        if len(opinions) > 0:
            ratio = min(opinions)/max(opinions)
            ratios.append(ratio)
    end = time.time()
    
    print("{}:\tAvg. -> {:0.6f}, Min. -> {:0.6f}, Max -> {:0.6f}, Std. -> {:0.6f}, Comp. time -> {:0.6f}".format(
        f.__name__, np.mean(ratios), min(ratios), max(ratios), np.std(ratios), end - start
    ))

uniform   = lambda: random.random(); uniform.__name__ = "uniform"
polarized = lambda: arcsine.rvs();   polarized.__name__ = "polarized"

for topo_code in ["eu-core"]: #["amazon", "dblp", "eu-core", "facebook", "twitter"]:
    print(f"Topology -> {topo_code}")
    print("\tCommunity-Agnostic:")
    for opinion_func in [uniform, polarized]:
        graph, comm = Simulation.load_graph_and_communities(topo_code)
        opinion = Simulation.initialize_opinion(graph, distr_func=opinion_func)
        lo = [x for x in opinion if x <  0.5]
        up = [x for x in opinion if x >= 0.5]
        print("\t * {}.lower:\tAvg={:0.6f}, Min={:0.6f}, Max={:0.6f}, Std={:0.6f}".format(
            opinion_func.__name__, np.mean(lo), min(lo), max(lo), np.std(lo)
        ))
        print("\t * {}.upper:\tAvg={:0.6f}, Min={:0.6f}, Max={:0.6f}, Std={:0.6f}".format(
            opinion_func.__name__, np.mean(up), min(up), max(up), np.std(up)
        ))

    print("\tCommunity-Aware:")
    for opinion_func in [uniform, polarized]:
        graph, comm = Simulation.load_graph_and_communities(topo_code)
        opinion = Simulation.initialize_opinion(graph, comm, distr_func=opinion_func)
        lo = [x for x in opinion if x <  0.5]
        up = [x for x in opinion if x >= 0.5]
        print("\t * {}.lower:\tAvg={:0.6f}, Min={:0.6f}, Max={:0.6f}, Std={:0.6f}".format(
            opinion_func.__name__, np.mean(lo), min(lo), max(lo), np.std(lo)
        ))
        print("\t * {}.upper:\tAvg={:0.6f}, Min={:0.6f}, Max={:0.6f}, Std={:0.6f}".format(
            opinion_func.__name__, np.mean(up), min(up), max(up), np.std(up)
        ))

"""
n_trials = 10
time_horizon = 100
default_algorithms = [
    TIM_plus_solution, LAIM_solution, fast_LAIM_solution,  # Baselines
    opinion_degree_solution, proto1_solution,              # Proposed
    degree_solution, min_opinion_solution, random_solution # Heuristics
]

seed_sizes = list(range(0, 50+1, 10))
seed_sizes[0] = 1
seed_sizes = [0] + seed_sizes

results = pd.DataFrame()
for alg in default_algorithms:
    filename = "alg={}".format(alg.__name__.replace("_solution", ""))
    results = results.append(Simulation.run(
        topo_code          = "facebook", 
        algorithm          = alg, 
        seed_sizes         = seed_sizes, 
        time_horizon       = time_horizon, 
        n_trials           = n_trials, 
        opinion_distr_func = lambda: 1.0,
        ffm_distr_func     = lambda: 1.0,
        random_seed        = 46578641,
        use_communities    = False,
        pbar_desc          = filename
    ))

results.to_csv("fb-playground-data.csv")
sns.barplot(x="seed_size", y="activated", hue="algorithm", data=results)
plt.show()
"""