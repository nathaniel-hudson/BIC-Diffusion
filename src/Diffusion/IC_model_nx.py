import Constants
import datetime
import matplotlib.pyplot as plt
import networkx as nx
import numpy    as np
import pandas   as pd
import random   as rd
import seaborn  as sns

from scipy.stats import arcsine, uniform

class IC_Model(object):
    def __init__(self, graph, pp=0.1):
        """Initialize an instance of the BIC_Model.

        Parameters
        ----------
        graph : nx.DiGraph
            Social network topology.
        ffm : np.array
            Two-dimensional array (n x 5) to represent the FFM values for each user node.
        init_opinion : np.array
            One-dimensional array to represent the initial opinions for each user node.
        """
        self.graph = graph
        self.pp    = pp


    def diffuse(self, active_set):
        """Performs a single diffusion step to be used in a simulation.

        Parameters
        ----------
        active_set : set
            Set of activated nodes (i.e., active spreaders) to influence opinion.

        Returns
        -------
        (active_set: set, visited: set)
            A tuple containing the the set of active/activated user nodes and user nodes that have been visited in total.
        """
        activated = set()
        
        for node in set(active_set):
            neighbors = set(self.graph.neighbors(node))
            for neighbor in neighbors - active_set:
                if rd.random() <= self.pp:
                    activated.add(neighbor)
                    active_set.add(neighbor)
                
        return activated

"""
Running a simulation using the IC model with a LARGE online social network topology.
This will also analyze the runtime/load time of the networks.
"""
if __name__ == '__main__':
    import time

    from Algorithms.Heuristics import random_solution, degree_solution
    from tqdm                  import tqdm

    data = {'k': [], '|A|': [], 'total-time': []}
    
    rd.seed(17)
    graph = nx.read_edgelist('topos/amazon-graph.txt')
    model = IC_Model_nx(graph)
    seed_sizes = list(range(0, 1001, 100)); seed_sizes[0] = 1

    for k in seed_sizes: 
        print(f'>> Running for seed size (k = {k}).')
        diff_start = time.time()
        active_set = set(degree_solution(model, k))
        for timestep in tqdm(range(100)):
            new_set = model.diffuse(active_set)
            if active_set == new_set:
                break
            else:
                active_set = new_set

        data['k'].append(k)
        data['|A|'].append(len(active_set))
        data['total-time'].append(time.time() - diff_start)

    df = pd.DataFrame.from_dict(data)
    
    sns.barplot(x='k', y='|A|', data=df)
    plt.title('Activtion Set vs. Seed Set Size')
    plt.show()
    plt.clf()

    sns.barplot(x='k', y='total-time', data=df)
    plt.title('Runtime vs. Seed Set Size')
    plt.show()
    plt.clf()