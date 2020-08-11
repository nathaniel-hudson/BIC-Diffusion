import networkx as nx
import random

from Diffusion.Model import BIC

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