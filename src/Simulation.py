import matplotlib.pyplot as plt
import networkx as nx
import numpy    as np
import random   as rd
import seaborn  as sns

from Model import *
from tqdm import tqdm

sns.set_style('darkgrid')

ALGORITHMS = {
    'random': lambda graph, k: rd.sample(graph.nodes, k)
}


def main():
    columns = ['trial', 'time_step', 'opinion', 'activated', 'algorithm', 'seed_size']

    data = {column: [] for column in columns}
    seed_set_sizes = list(range(1, 10+1))
    n_trials = 2
    time_horizon = 100

    for trial in tqdm(range(n_trials)):
        # graph = nx.watts_strogatz_graph(300, 10, 0.2)
        graph = nx.karate_club_graph()
        opinion = np.array([rd.random() for user in graph.nodes])
        ffm = {user: {factor: rd.random() for factor in 'OCEAN'} for user in graph.nodes}
        model = BIC_Model(graph, ffm, opinion)

        for n_seeds in seed_set_sizes:
            model.prepare()
            for alg_label in ALGORITHMS:
                algorithm = ALGORITHMS[alg_label]
                active_set = algorithm(graph, n_seeds)
                for time_step in range(time_horizon):
                    ## Perform diffusion step.
                    active_set = model.diffuse(active_set, time_step)

                    ## Add record to the data.
                    data['trial'].append(trial)
                    data['time_step'].append(time_step)
                    data['opinion'].append(sum(model.opinion[time_step]))
                    data['activated'].append(len(active_set))
                    data['algorithm'].append(alg_label)
                    data['seed_size'].append(n_seeds)

    ## Convert data dictionary to a pandas DataFrame.
    df = pd.DataFrame.from_dict(data)
    sns.barplot(x='seed_size', y='opinion', hue='algorithm', 
                data=df.query(f'time_step == {time_horizon-1}'))
    plt.show()
    

if __name__ == '__main__':
    main()