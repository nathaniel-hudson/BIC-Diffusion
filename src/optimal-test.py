import Algorithms
import BIC_NK
import matplotlib.pyplot as plt
import networkit as nk
import pandas    as pd
import progressbar
import random    as rd
import seaborn   as sns
import time
import warnings

sns.set_style('darkgrid')
warnings.filterwarnings('ignore')

df = pd.DataFrame()

n_nodes = 20
p = 0.125
budget = 1
n_iters = 5
n_trials = 20
t_horizon = 10
status = progressbar.ProgressBar(n_iters).start()

start = time.time()
for i in range(n_iters):
    # Update the status.
    status.update(i)

    # Generate the random network topology.
    gen = nk.generators.ErdosRenyiGenerator(n_nodes, p, directed=True)
    graph = gen.generate()
    opinion = {node: {0: rd.random()} for node in graph.nodes()}
    ffm = {node: {'O': rd.random(), 'C': rd.random(), 'E': rd.random(),
                'A': rd.random(), 'N': rd.random()} for node in graph.nodes()}
    
    # Collect the seed sets.
    degree_seeds = Algorithms.degree(graph, budget)
    greedy_seeds = Algorithms.greedy_OM(graph, budget, opinion, ffm, t_horizon, 15)
    o_path_seeds = Algorithms.opinion_path(graph, budget, opinion, ffm)

    # Run simulations.
    df = df.append(BIC_Model.simulate(graph, greedy_seeds,
                                      opinion, ffm, t_horizon, n_trials, 'Degree'))
    df = df.append(BIC_Model.simulate(graph, greedy_seeds,
                                      opinion, ffm, t_horizon, n_trials, 'Greedy'))
    df = df.append(BIC_Model.simulate(graph, greedy_seeds,
                                      opinion, ffm, t_horizon, n_trials, 'OpinionPath'))
end = time.time()
status.update(n_iters)

sns.lineplot(y='opinion', x='time-step', data=df,
             hue='algorithm', marker='.')
plt.title('Opinion Evolution -- %f sec.' % (end-start))
plt.show()

sns.lineplot(y='activated', x='time-step',
             data=df, hue='algorithm', marker='.')
plt.title('Activation Evolution -- %f sec.' % (end-start))
plt.show()
