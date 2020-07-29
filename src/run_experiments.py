import numpy  as np
import pandas as pd
import random as rd

TOPOS = {
    'com-Amazon': ## Pick.
    {
        'edgelist':  'com-amazon.ungraph.txt.gz',
        'community': 'com-amazon.top5000.cmty.txt.gz',
        'directed':  False
    },
    'com-DBLP': ## Pick.
    {
        'edgelist':  'com-dblp.ungraph.txt.gz',
        'community': 'com-dblp.top5000.cmty.txt.gz',
        'directed':  False
    },
    'com-EU-core': ## Pick.
    {
        'edgelist':  'com-Eu-core.txt.gz',
        'community': 'com-Eu-core-department-labels.txt.gz',
        'directed':  True
    },
    'com-LiveJournal': 
    {
        'edgelist':  'com-lf.ungraph.txt.gz',
        'community': 'com-lj.top5000.cmty.txt.gz',
        'directed':  False
    },
    'com-Orkut':
    {
        'edgelist':  'com-orkut.ungraph.txt.gz',
        'community': 'com-orkut.top5000.cmty.txt.gz',
        'directed':  False
    },
    'com-YouTube':
    {
        'edgelist':  'com-youtube.ungraph.txt.gz',
        'community': 'com-youtube.top5000.cmty.txt.gz',
        'directed':  False
    },
    'wiki-topcats': ## Pick.
    {
        'edgelist':  'wiki-topcats.txt.gz',
        'community': 'wiki-topcats-categories.txt.gz',
        'directed':  True
    },
}

def load_network(topo_code):
    pass

def community_agnostic_opinionate(graph, communities, opinion_fn):
    pass

def community_aware_opinionate(graph, communities, opinion_fn):
    pass