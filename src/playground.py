import matplotlib.pyplot as plt
import networkx as nx
import random
import seaborn as sns
import time

import Simulation

from Algorithms.Heuristics import *
from Algorithms.LAIM       import *
from Algorithms.TIM        import TIM_solution
from Algorithms.TIM_Plus   import TIM_plus_solution

from Diffusion.Model import BIC

"""
Using the new implementation of TIM+... On a BA network with 1000 nodes and K=32, we have the following runtimes and we
consider n_seeds=5:
    TIM+ runtime      746.75121
    LAIM runtime      57.85623
    FastLAIM runtime  2.84473
"""

sns.set_style("darkgrid")

n_trials = 5#50
n_seeds = 50
# seed_sizes = list(range(0, 100+1, 10))
# seed_sizes[0] = 1
time_horizon = 100
uniform = random.random

n = 15000
m = 31000

graph = nx.gnm_random_graph(1500000, 31000000, directed=False) ## NOTE: This emulates the NETHEPT topology's density.
ffm = {node: {factor: random.random() for factor in "OCEAN"} for node in graph.nodes()}
opinion = [random.random() for node in graph.nodes()]

model = BIC(graph, ffm, opinion)
model.prepare()

start = time.time()
seed_set = TIM_plus_solution(model, n_seeds)
TIM_time = time.time() - start

# start = time.time()
# seed_set = LAIM_solution(model, n_seeds)
# LAIM_time = time.time() - start

start = time.time()
seed_set = fast_LAIM_solution(model, n_seeds)
fast_LAIM_time = time.time() - start

print(f"\n\nTIM+ runtime:     {TIM_time:0.5f}")
# print(f"LAIM runtime:     {LAIM_time:0.5f}")
print(f"FastLAIM runtime: {fast_LAIM_time:0.5f}")

# model.prepare()
# opinion, activated, visited = model.simulate(seed_set, 100)

# print("Total Opinion: {}\nActivated nodes: {}\nVisited nodes: {}\nLAIM runtime: {:0.5f}".format(
#     opinion, len(activated), len(visited), end - start
# ))
