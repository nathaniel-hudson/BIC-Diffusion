import datetime
import gzip
import matplotlib.pyplot as plt
import networkx as nx
import numpy    as np
import os
import pandas   as pd
import random
import seaborn  as sns
import time

from Diffusion.Model import BIC
from tqdm            import tqdm
from tqdm            import tqdm_notebook as tqdm_note

COLUMNS = ["trial", "opinion", "activated", "algorithm", "seed_size", "algorithm_time"]

def get_relabel_mapping(graph):
    node_idx = 0
    mapping  = {}
    for node in graph.nodes():
        mapping[node] = node_idx
        node_idx += 1
    return mapping



def initialize_ffm_factors(graph, distr_func=random.random):
    return {node: {factor: distr_func() for factor in "OCEAN"} for node in graph.nodes()}


def load_communities(comm_path, mapping=None):
    communities = {}
    with gzip.open(comm_path, "r") as f:
        i = 0
        if mapping is None:
            for line in f:
                communities[i] = [int(node) for node in line.split()]
                i += 1
        else:
            for line in f:
                try:
                    communities[i] = [mapping[int(node)] for node in line.split()]
                    i += 1
                except KeyError as err:
                    print([int(node) for node in line.split()])
                    print(mapping.keys())
                    print(err)
                    exit(-1)

    return communities


def initialize_opinion(topo_code, graph, communities=None, distr_func=random.random):
    if communities is None:
        opinion = [distr_func() for node in graph.nodes()]
    else:
        opinion = [distr_func() for node in graph.nodes()]

        node_opinion = {node: [] for node in graph.nodes()}
        comm_opinion = {comm_idx: distr_func() for comm_idx in communities}
        for comm_idx in communities:
            for node in communities[comm_idx]:
                node_opinion[node].append(comm_opinion[comm_idx])

        for node in graph.nodes():
            if len(node_opinion[node]) > 0:
                opinion[node] = np.mean(node_opinion[node])

    return opinion

def load_graph(topo_code):
    base_path = os.path.join("..", "topos")
    if topo_code == "amazon":
        graph_type = nx.Graph
        graph_path = os.path.join(base_path, "com-amazon.ungraph.txt.gz")
        comm_path  = os.path.join(base_path, "com-amazon.top5000.cmty.txt.gz")
    elif topo_code == "dblp":
        graph_type = nx.Graph
        graph_path = os.path.join(base_path, "com-dblp.ungraph.txt.gz")
        comm_path  = os.path.join(base_path, "com-dblp.top5000.cmty.txt.gz")
    elif topo_code == "eu-core":
        graph_type = nx.DiGraph
        graph_path = os.path.join(base_path, "email-Eu-core.txt.gz")
        comm_path  = os.path.join(base_path, "email-Eu-core-department-labels.txt.gz")
    elif topo_code == "lj":
        graph_type = nx.Graph
        graph_path = os.path.join(base_path, "com-lj.ungraph.txt.gz")
        comm_path  = os.path.join(base_path, "com-lj.top5000.cmty.txt.gz")
    elif topo_code == "orkut":
        graph_type = nx.Graph
        graph_path = os.path.join(base_path, "com-orkut.ungraph.txt.gz")
        comm_path  = os.path.join(base_path, "com-orkut.top5000.cmty.txt.gz")
    elif topo_code == "youtube":
        graph_type = nx.Graph
        graph_path = os.path.join(base_path, "com-youtube.ungraph.txt.gz")
        comm_path  = os.path.join(base_path, "com-youtube.top5000.cmty.txt.gz")
    elif topo_code == "wiki":
        graph_type = nx.DiGraph
        graph_path = os.path.join(base_path, "wiki-topcats.txt.gz")
        comm_path  = os.path.join(base_path, "wiki-topcats-categories.txt.gz")
    else:
        raise ValueError("Invalid value for parameter `topo_code`.")
    
    # Initialize the graph the FFM factors.
    graph = nx.read_edgelist(graph_path, create_using=graph_type, nodetype=int)
    return graph, comm_path


def run(topo_code, algorithm, seed_sizes, time_horizon, n_trials, ffm_distr_func, opinion_distr_func, 
        threshold=5, random_seed=None, use_communities=False, out_dir=None, mode="console"):
    """Run an experiment using a specific algorithm and a topology. The parameters tightly define the experimental setup 
       in a self-explanatory fashion. The function outputs the results into the "../out/results/`out_dir`" directory 
       with a .CSV file dedicated to the results for this one result. The resulting CSV files can be merged using Pandas 
       API to compare the results across algorithms.

    Parameters
    ----------
    topo_code : str
        The code for the real-world topology to be considered.
    algorithm : func
        Pointer to the function that you wish to run --- must take a BIC model and int (for # seed) as input and return 
        a set.
    seed_sizes : list/set
        Iterable object containing the sizes of seed sets to be considered.
    time_horizon : int
        Number of time-steps to be considered.
    n_trials : int
        Number of Monte-Carlo runs to be considered.
    ffm_distr_func : func
        Random generator for FFM factors.
    opinion_distr_func : func
        Random generator for opinion values.
    random_seed : int, optional
        Random seed, if provided, to ensure fair comparisons (called before each graph is instantiated), by default
        None.
    use_communities : bool, optional
        True if you want to consider community opinion initialization, False otherwise, by default False.
    out_dir : str, None
        Output directory for the results of this experiment if provided (should be standard across algorithms for a set 
        of experiments); do not save data if None, by default None.
    mode : str, "console"
        If the value of `mode` is "notebook" then use the appropriate `tqdm` API, by default "console".
    """
    data = {column: [] for column in COLUMNS}
    if mode == "notebook":
        pbar = tqdm_note(total=n_trials * len(seed_sizes))
    else:
        pbar = tqdm(total=n_trials * len(seed_sizes))

    graph, comm_path = load_graph(topo_code)
    mapping = get_relabel_mapping(graph)
    graph = nx.relabel_nodes(graph, mapping)
    if use_communities:
        communities = load_communities(comm_path, mapping)
    else:
        communities = None

    for n_seeds in seed_sizes:
        if random_seed is not None: random.seed(random_seed)

        for trial in range(n_trials):
            # Initialize the FFM factors and the opinion vector.
            ffm = initialize_ffm_factors(graph, distr_func=ffm_distr_func)
            opinion = initialize_opinion(topo_code, graph, communities, distr_func=opinion_distr_func)
            
            # Prepare the model, run seed selection algorithm, and run simulation.
            model = BIC(graph, ffm, opinion)
            start_time = time.time()
            seed_set = algorithm(model, n_seeds)
            alg_runtime = time.time() - start_time
            model.prepare(threshold=threshold)
            total_opinion, activated, visited = model.simulate(seed_set, time_horizon)

            # Add record to the data.
            data["trial"].append(trial)
            data["opinion"].append(total_opinion)
            data["activated"].append(len(activated))
            data["algorithm"].append(algorithm.__name__)
            data["seed_size"].append(n_seeds)
            data["algorithm_time"].append(alg_runtime)

            # Update the progress bar.
            pbar.set_description(f"{algorithm.__name__}, K = {n_seeds}, Trial ({trial+1}/{n_trials})")
            pbar.update(1)

    ## Convert data dictionary to a pandas DataFrame and save.
    df = pd.DataFrame.from_dict(data)
    if out_dir is not None:
        path = os.path.join("..", "out", "results", out_dir)
        if not os.path.exists(path):
            os.makedirs(path)
        df.to_csv(os.path.join(path, f"{algorithm.__name__}-{topo_code}-values.csv"))
    return df