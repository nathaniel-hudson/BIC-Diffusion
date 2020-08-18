import networkx as nx
import random
import time

import Simulation

from Algorithms.Baselines import *
from Diffusion.Model import BIC

n_trials = 50
seed_sizes = list(range(0, 100+1, 10))
seed_sizes[0] = 1
time_horizon = 100
topo_code = "eu-core" #"amazon"
uniform = random.random

TIM_start = time.time()
df_TIM = Simulation.run(
    topo_code=topo_code, 
    algorithm=TIM_solution, 
    seed_sizes=seed_sizes, 
    time_horizon=time_horizon, 
    n_trials=n_trials, 
    ffm_distr_func=uniform,
    opinion_distr_func=uniform,
    random_seed=None,#random_seed, 
    use_communities=False,
    out_dir=None
)
TIM_runtime = time.time() - TIM_start
print(f"\n<> Simulation using `{TIM_solution.__name__}` took {TIM_runtime:0.5f} seconds.")

exit(0)


graph = nx.barabasi_albert_graph(1005, 20)
seeds = set(random.sample(graph.nodes(), k=100))
ffm = {node: {factor: random.random() for factor in "OCEAN"} for node in graph.nodes()}
opinion = [random.random() for node in graph.nodes()]
t_horizon = 100

model = BIC(graph, ffm, opinion)
model.prepare(threshold=10)

template = "Number of active nodes: {}, Number of killed nodes: {}, Number of unactivated nodes: {}, Total opinion: {}"
active_set = seeds
killed_set = set()
for tstep in range(t_horizon):
    print(template.format(
        len(active_set),  len(killed_set),  len(graph) - len(active_set.union(killed_set)),  
        model.total_opinion()
    ))
    active_set, killed_set = model.diffuse(active_set, killed_set, tstep)