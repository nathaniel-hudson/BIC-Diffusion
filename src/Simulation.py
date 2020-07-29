import matplotlib.pyplot as plt
import networkx as nx
import numpy    as np
import os
import progressbar
import random   as rd
import seaborn  as sns
import time

from BIC_model_nx import *
from tqdm         import tqdm

from Algorithms import Heuristics, Proposed # Baselines

sns.set_style('whitegrid')

ALGORITHMS = {
    # 'IMM':         Baselines.IMM_solution,
    # 'LAIM':        Baselines.LAIM_solution,
    # 'TIM+':        Baselines.TIM_solution,
    'OpinionDeg':  Proposed.opinion_degree_solution,
    # 'OpinionPath': Proposed.opinion_path_solution,
    # 'Degree':      Heuristics.degree_solution,
    'Min-Opinion': Heuristics.min_opinion_solution,
    'Random':      Heuristics.random_solution,
}
MARKERS = 'oPhsD*XH'

def main(random_seed):
    rd.seed(random_seed)

    columns = ['trial', 'opinion', 'activated', 'num_visited', 'algorithm', 'seed_size', 'algorithm time', 'load time']

    data = {column: [] for column in columns}
    seed_set_sizes = list(range(0, 1000+1, 100)); seed_set_sizes[0] = 1
    n_trials = 1 # 20
    time_horizon = 100

    pbar = progressbar.ProgressBar(maxval=n_trials*len(seed_set_sizes)).start()
    i = 0

    for trial in tqdm(range(n_trials)):
        load_start_time = time.time()
        graph = nx.read_edgelist(os.path.join('topos', 'amazon-graph.txt'), nodetype=int)
        opinion = np.array([rd.random() for user in graph.nodes])
        ffm = {user: {factor: rd.random() for factor in 'OCEAN'} for user in graph.nodes}
        model = BIC_Model_nx(graph, ffm, opinion)
        load_time = time.time() - load_start_time

        for n_seeds in seed_set_sizes:
            model.prepare()
            for alg_label in ALGORITHMS:
                ## Grab and run the algorithm while recording its runtime.
                algorithm   = ALGORITHMS[alg_label]
                start_time  = time.time()
                active_set  = algorithm(model, n_seeds)
                killed_set  = set()
                alg_runtime = time.time() - start_time

                ## Perform diffusion steps for the simulation.
                for time_step in range(time_horizon+1):
                    active_set, killed_set = model.diffuse(active_set, killed_set, time_step)

                ## Add record to the data.
                data['trial'].append(trial)
                data['opinion'].append(model.total_opinion())
                data['activated'].append(len(active_set.union(killed_set))) # len(active_set))
                # data['num_visited'].append(len(visited))
                data['algorithm'].append(alg_label)
                data['seed_size'].append(n_seeds)
                data['algorithm time'].append(alg_runtime)
                data['load time'].append(load_time)

                ## Update the progress bar.
                i += 1
                pbar.update(i)


    ## Convert data dictionary to a pandas DataFrame.
    df = pd.DataFrame.from_dict(data)
    sns.barplot(x='seed_size', y='opinion', hue='algorithm', data=df)
    plt.show(); plt.clf()

    sns.barplot(x='seed_size', y='algorithm time', hue='algorithm', data=df)
    plt.show(); plt.clf()

    df.to_csv(os.path.join('out', f'results-{time.time()}.csv'))
    

if __name__ == '__main__':
    main(7)