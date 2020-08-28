"""
This file is straight-forward, it simply runs the experiments and saves the data.
Experiments are run (and separate data files are saved) for all combinations of the 
following parameters:
    (1) Topology
    (2) Algorithm solution
    (3) Opinion distribution function (uniform/polarized)
    (4) Community-aware or Community-agnostic?
"""

import itertools
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import random
import Simulation
import time
import networkx as nx

from Algorithms.LAIM       import *
from Algorithms.TIM_Plus   import TIM_plus_solution
from Algorithms.Proposed   import *
from Algorithms.Heuristics import *
from Diffusion.Model       import *
from scipy.stats           import arcsine
from tqdm                  import tqdm

sns.set_style("darkgrid")
RESULTS_PATH = os.path.join("..", "out", "results")
META_PATH = os.path.join(RESULTS_PATH, "meta.json")

uniform   = lambda: random.random(); uniform.__name__ = "uniform"
polarized = lambda: arcsine.rvs();   polarized.__name__ = "polarized"

# Dependent variables.
default_topologies = ["amazon", "dblp", "eu-core", "facebook", "twitter"]
default_algorithms = [
    TIM_plus_solution, LAIM_solution, fast_LAIM_solution, # Baselines
    opinion_degree_solution,                              # Proposed
    degree_solution, IRIE_solution, min_opinion_solution  # Heuristics
]
seed_sizes = list(range(0, 50+1, 10)); seed_sizes[0] = 1
opinion_distrs = [uniform, polarized]
use_communities = [False, True]
alg_codes = {
    "TIM+": TIM_plus_solution,
    "LAIM": LAIM_solution,
    "fast_LAIM": fast_LAIM_solution,
    "new": new_solution,
    "opinion_degree": opinion_degree_solution,
    "degree": degree_solution,
    "IRIE": IRIE_solution,
    "min_opinion": min_opinion_solution
}

# Constant parameters.
n_trials = 100#0
t_horizon = 100
random_seed = 16785407 # a large *prime* number

if __name__ == "__main__":
    import argparse

    ## Use the argparse class to extract passed-in arguments from the command line. This
    ## is done to simplify the process of running multiple algorithms in parallel without
    ## explicit multi-threading.
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg")
    parser.add_argument("--topo")
    args = parser.parse_args()

    if args.alg is None:
        algorithms = default_algorithms
    else:
        alg_func = alg_codes[args.alg]
        algorithms = [alg_func]
    if args.topo is None:
        topologies = default_topologies
    else:
        if args.topo not in Simulation.TOPO_CODES:
            raise ValueError("Invalid value for arg `topo`.")
        topologies = [args.topo]

    ## Run the experiments over all combinations of the paramters.
    param_combinations = itertools.product(topologies, algorithms, opinion_distrs, use_communities)
    for (topo_code, alg, opinion_distr, use_comm) in param_combinations:
        filename = "topo={}_alg={}_comm={}_{}".format(
            topo_code, alg.__name__.replace("_solution", ""), use_comm, 
            opinion_distr.__name__
        )
        results = Simulation.run(
            topo_code          = topo_code, 
            algorithm          = alg, 
            seed_sizes         = seed_sizes, 
            time_horizon       = t_horizon, 
            n_trials           = n_trials, 
            opinion_distr_func = opinion_distr,
            random_seed        = random_seed,
            use_communities    = use_comm,
            pbar_desc          = filename
        )
        results.to_csv(os.path.join(RESULTS_PATH, filename + ".csv"))
