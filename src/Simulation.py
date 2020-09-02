import datetime
import gzip
import matplotlib.pyplot as plt
import networkx as nx
import numpy    as np
import os, sys
import pandas   as pd
import random
import seaborn  as sns
import time

from Diffusion.Model import BIC
from tqdm            import tqdm
from tqdm            import tqdm_notebook as tqdm_note

COLUMNS = [
    "trial", "opinion", "activated", "visited", "algorithm", "seed_size", 
    "algorithm_time", "opinion_distr_func", "ffm_distr_func", "use_communities"
]

TOPO_CODES = [
    "amazon", "dblp", "eu-core", "facebook", "gplus", "lj", "orkut", "twitter", "youtube", 
    "wiki"
]


def get_network_loading_params(topo_code):
    """This function is used to retrieve the necessary data/params in order to load a 
       network topology. This function will provide params regarding the graph-type 
       (directed or undirected), the path to the network topology, the path to the 
       community file, and a lambda function that can be used to retrieve the community 
       dictionary.

    Parameters
    ----------
    topo_code : str
        Code associated with a real-world network topology.

    Returns
    -------
    (class, str, str, func)
        A 4-tuple containing the following items: (1) graph class object of either 
        nx.Graph or nx.DiGraph, (2) path to network topology, (3) path to community file, 
        and (4) function to parse community into dictionary.

    Raises
    ------
    ValueError
        Raised if an invalid `topo_code` is provided.
    """
    base_path = os.path.join("topos")
    if topo_code == "amazon":
        graph_type = nx.Graph
        graph_path = os.path.join(base_path, "com-amazon.ungraph.txt.gz")
        comm_path  = os.path.join(base_path, "com-amazon.top5000.cmty.txt.gz")
        comm_func  = lambda comm_path, mapping: \
            load_community_by_line(comm_path, mapping, n_line_prefix=0)

    elif topo_code == "dblp":
        graph_type = nx.Graph
        graph_path = os.path.join(base_path, "com-dblp.ungraph.txt.gz")
        comm_path  = os.path.join(base_path, "com-dblp.top5000.cmty.txt.gz")
        comm_func  = lambda comm_path, mapping: \
            load_community_by_line(comm_path, mapping, n_line_prefix=0)

    elif topo_code == "eu-core":
        graph_type = nx.DiGraph
        graph_path = os.path.join(base_path, "email-Eu-core.txt.gz")
        comm_path  = os.path.join(base_path, "email-Eu-core-department-labels.txt.gz")
        comm_func  = lambda comm_path, mapping: \
            load_community_by_pairs(comm_path, mapping)

    elif topo_code == "facebook":
        graph_type = nx.Graph
        graph_path = os.path.join(base_path, "facebook_combined.txt.gz")
        comm_path  = os.path.join(base_path, "com-facebook.all.cmty.txt.gz")
        comm_func  = lambda comm_path, mapping: \
            load_community_by_line(comm_path, mapping, n_line_prefix=0)


    elif topo_code == "gplus":
        graph_type = nx.DiGraph
        graph_path = os.path.join(base_path, "gplus_combined.txt.gz")
        comm_path  = os.path.join(base_path, "com-gplus.all.cmty.txt.gz")
        comm_func  = lambda comm_path, mapping: \
            load_community_by_line(comm_path, mapping, n_line_prefix=0)


    elif topo_code == "lj":
        graph_type = nx.Graph
        graph_path = os.path.join(base_path, "com-lj.ungraph.txt.gz")
        comm_path  = os.path.join(base_path, "com-lj.top5000.cmty.txt.gz")
        comm_func  = lambda comm_path, mapping: \
            load_community_by_line(comm_path, mapping, n_line_prefix=0)

    elif topo_code == "orkut":
        graph_type = nx.Graph
        graph_path = os.path.join(base_path, "com-orkut.ungraph.txt.gz")
        comm_path  = os.path.join(base_path, "com-orkut.top5000.cmty.txt.gz")
        comm_func  = lambda comm_path, mapping: \
            load_community_by_line(comm_path, mapping, n_line_prefix=0)

    elif topo_code == "twitter":
        graph_type = nx.DiGraph
        graph_path = os.path.join(base_path, "twitter_combined.txt.gz")
        comm_path  = os.path.join(base_path, "com-twitter.all.cmty.txt.gz")
        comm_func  = lambda comm_path, mapping: \
            load_community_by_line(comm_path, mapping, n_line_prefix=0)


    elif topo_code == "youtube":
        graph_type = nx.Graph
        graph_path = os.path.join(base_path, "com-youtube.ungraph.txt.gz")
        comm_path  = os.path.join(base_path, "com-youtube.top5000.cmty.txt.gz")
        comm_func  = lambda comm_path, mapping: \
            load_community_by_line(comm_path, mapping, n_line_prefix=0)

    elif topo_code == "wiki":
        graph_type = nx.DiGraph
        graph_path = os.path.join(base_path, "wiki-topcats.txt.gz")
        comm_path  = os.path.join(base_path, "wiki-topcats-categories.txt.gz")
        comm_func  = lambda comm_path, mapping: \
            load_community_by_line(comm_path, mapping, n_line_prefix=1)

    else:
        raise ValueError("Invalid value for parameter `topo_code`.")

    return graph_type, graph_path, comm_path, comm_func


def get_relabel_mapping(graph):
    node_idx = 0
    mapping  = {}
    for node in graph.nodes():
        mapping[node] = node_idx
        node_idx += 1
    return mapping


def initialize_ffm_factors(graph, distr_func=random.random):
    return {node: {factor: distr_func() for factor in "OCEAN"} for node in graph.nodes()}


def load_community_by_line(comm_path, mapping=None, n_line_prefix=0):
    comms = {}
    with gzip.open(comm_path, "r") as f:
        i = 0
        if mapping is None:
            for line in f:
                comms[i] = [int(node) for node in line.split()[n_line_prefix:]]
                i += 1
        else:
            for line in f:
                comms[i] = [mapping[int(node)] for node in line.split()[n_line_prefix:]]
                i += 1

    return comms


def load_community_by_pairs(comm_path, mapping=None):
    comms = {}
    with gzip.open(comm_path, "r") as f:
        for line in f:
            node_idx, comm_idx = line.split()
            node_idx, comm_idx = int(node_idx), int(comm_idx)
            if mapping is not None:
                node_idx = mapping[node_idx]

            if comm_idx not in comms:
                comms[comm_idx] = []

            comms[comm_idx].append(node_idx)
    
    return comms    


def initialize_opinion(graph, communities=None, distr_func=random.random):
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
    graph_type, graph_path, comm_path, comm_func = get_network_loading_params(topo_code)
    graph = nx.read_edgelist(graph_path, create_using=graph_type, nodetype=int)
    mapping = get_relabel_mapping(graph)
    graph = nx.relabel_nodes(graph, mapping)
    return graph


def load_graph_and_communities(topo_code):
    graph_type, graph_path, comm_path, comm_func = get_network_loading_params(topo_code)
    graph = nx.read_edgelist(graph_path, create_using=graph_type, nodetype=int)
    mapping = get_relabel_mapping(graph)
    graph = nx.relabel_nodes(graph, mapping)
    communities = comm_func(comm_path, mapping)
    return graph, communities
    

def run(topo_code, algorithm, seed_sizes, time_horizon, n_trials, opinion_distr_func, 
        ffm_distr_func=random.random, threshold=5, random_seed=None, use_communities=False,
        out_dir=None, mode="console", pbar_desc=None):
    """Run an experiment using a specific algorithm and a topology. The parameters tightly 
       define the experimental setup  in a self-explanatory  fashion. The function outputs 
       the  results  into the  "../out/results/`out_dir`"  dir with  a .CSV file dedicated
       to  the results  for  this one  result. The resulting CSV files can be merged using 
       Pandas API to compare the results across algorithms.

    Parameters
    ----------
    topo_code : str
        The code for the real-world topology to be considered.
    algorithm : func
        Pointer  to the function  that you  wish to run --- must  take a BIC model and int 
        (for # seed) as input and return a set.
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
        Random seed, if provided, to ensure fair comparisons (called before each graph is 
        instantiated), by default None.
    use_communities : bool, optional
        True if you want to consider community opinion initialization, False otherwise, by
        default False.
    out_dir : str, None
        Output  directory for the results of this experiment if provided (should be stand-
        ard across  algorithms  for a  set of experiments); do not  save data  if None, by 
        default None.
    pbar_desc : str, "None"
        The description used for the progress bar via `tqdm` if not None, by default None.
    """
    data = {column: [] for column in COLUMNS}
    pbar = tqdm(total=n_trials*len(seed_sizes), file=sys.stdout, desc=pbar_desc)

    if use_communities:
        graph, communities = load_graph_and_communities(topo_code)
    else:
        graph, communities = load_graph(topo_code), None

    for n_seeds in seed_sizes:
        # Initialize the FFM factors and the opinion vector.
        if random_seed is not None: 
            random.seed(random_seed)
        ffm = initialize_ffm_factors(graph, distr_func=ffm_distr_func)
        opinion = initialize_opinion(graph, communities, distr_func=opinion_distr_func)
        
        # Prepare the model, run seed selection algorithm, and run simulation.
        model = BIC(graph, ffm, opinion)
        start_time = time.time()
        seed_set = algorithm(model, n_seeds)
        alg_runtime = time.time() - start_time
        random.seed(None) # NOTE: We only want to fix the FFM and opinion generation.

        for trial in range(n_trials):
            # Add record to the data.
            model.prepare(threshold=threshold)
            total_opinion, activated, visited = model.simulate(seed_set, time_horizon)
            data["trial"].append(trial)
            data["opinion"].append(total_opinion)
            data["activated"].append(len(activated))
            data["visited"].append(len(visited))
            data["algorithm"].append(algorithm.__name__)
            data["seed_size"].append(n_seeds)
            data["algorithm_time"].append(alg_runtime)
            data["opinion_distr_func"].append(opinion_distr_func.__name__)
            data["ffm_distr_func"].append(ffm_distr_func.__name__)
            data["use_communities"].append(use_communities)
            pbar.update(1)

    ## Convert data dictionary to a pandas DataFrame and save.
    df = pd.DataFrame.from_dict(data)
    if out_dir is not None:
        path = os.path.join("..", "out", "results", out_dir)
        if not os.path.exists(path):
            os.makedirs(path)
        filename = "{}-{}-values.csv"
        df.to_csv(os.path.join(path, filename.format(algorithm.__name__, topo_code)))
    return df