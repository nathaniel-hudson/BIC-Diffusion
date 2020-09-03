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
from tqdm import tqdm

uniform   = lambda: random.random(); uniform.__name__ = "uniform"
polarized = lambda: arcsine.rvs();   polarized.__name__ = "polarized"
data = {
    "Topology": [],
    "Opinion-Kind": [],
    "Community-Kind": [],
    "Metric": []
}

def get_metric(graph, opinion):
    stds = []
    # neighbor_func = graph.neighbors if not nx.is_directed(graph) else graph.predecessors
    directed = nx.is_directed(graph)
    for u in graph.nodes():
        ## OPTION A.
        # diffs = []
        # for v in neighbor_func(u):
        #     diff = abs(opinion[u] - opinion[v])
        #     diffs.append(diff)
        ## OPTION B.
        diffs = [opinion[u]] + [opinion[v] for v in graph.neighbors(u)]
        if directed:
            diffs.extend([opinion[v] for v in graph.predecessors(u)])
        if len(diffs) > 0:
            stds.append(np.std(diffs))
        else:
            stds.append(0)
    return stds

pbar = tqdm(total=5*2*2, desc="Progress")

for topo_code in ["amazon", "dblp", "eu-core", "facebook", "twitter"]:

    for opinion_func in [uniform, polarized]:
        graph, comm = Simulation.load_graph_and_communities(topo_code)
        opinion = Simulation.initialize_opinion(graph, distr_func=opinion_func)
        metrics = get_metric(graph, opinion)
        for metric in metrics:
            data["Topology"].append(topo_code)
            data["Opinion-Kind"].append(opinion_func.__name__)
            data["Community-Kind"].append("agnostic")
            data["Metric"].append(metric)
        pbar.update(1)
        
    for opinion_func in [uniform, polarized]:
        graph, comm = Simulation.load_graph_and_communities(topo_code)
        opinion = Simulation.initialize_opinion(graph, comm, distr_func=opinion_func)
        metrics = get_metric(graph, opinion)
        for metric in metrics:
            data["Topology"].append(topo_code)
            data["Opinion-Kind"].append(opinion_func.__name__)
            data["Community-Kind"].append("aware")
            data["Metric"].append(metric)
        pbar.update(1)
        

sns.catplot(
    x="Topology",
    y="Metric",
    hue="Topology",
    kind="boxen",
    data=pd.DataFrame.from_dict(data),
    row="Opinion-Kind",
    col="Community-Kind",
    sharex=True,
    sharey=True,
)
plt.show()


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