import matplotlib.pyplot as plt
import networkx as nx
import random
import seaborn as sns
import time

import Simulation

from Algorithms.Baselines import *
from Algorithms.Heuristics import * 

from Diffusion.Model import BIC

sns.set_style("darkgrid")

n_trials = 5#50
seed_sizes = list(range(0, 100+1, 10))
seed_sizes[0] = 1
time_horizon = 100
uniform = random.random


#graph = nx.karate_club_graph()
graph = nx.barabasi_albert_graph(100, 6)
ffm = {node: {factor: random.random() for factor in "OCEAN"} for node in graph.nodes()}
opinion = [random.random() for node in graph.nodes()]

model = BIC(graph, ffm, opinion)
start = time.time()
seed_set = TIM_solution(model, 5)
end = time.time()
model.prepare()
opinion, activated, visited = model.simulate(seed_set, 100)

print("Total Opinion: {}\nActivated nodes: {}\nVisited nodes: {}\nTIM+ runtime: {:0.5f}".format(
    opinion, len(activated), len(visited), end - start
))

exit(0)

sns.lineplot(
    x="seed_size", y="opinion", style="algorithm", hue="algorithm", markers=True, err_style="bars", data=results
)
plt.title("TIM+ Solution: Opinion")
plt.show()

sns.lineplot(
    x="seed_size", y="activated", style="algorithm", hue="algorithm", markers=True, err_style="bars",data=results
)
plt.title("TIM+ Solution: Activation")
plt.show()

print(results.head())