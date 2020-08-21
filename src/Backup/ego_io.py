import glob
import networkx as nx
import numpy as np
import os.path
import random


def get_circles(directory, mapping=None):
    """Read in the circles for the topology under consideration. This function will grab each
       *.circles file and create a dictionary with the nodes that each circle contains.
    
    Arguments:
        directory {str} -- The name/path of the directory to where the *.circles files are located.
    """
    circles = {}
    paths = glob.glob(os.path.join(directory, '*.circles'))
    for p in paths:
        with open(p, 'r') as f:
            for line in f:
                line = line.split()
                circle_id = line[0]
                node_ids = line[1:]
                while circle_id in circles:
                    circle_id += '_'
                if mapping is None:
                    circles[circle_id] = [int(node) for node in node_ids]
                else:
                    circles[circle_id] = [mapping[int(node)] for node in node_ids]
    return circles


def get_ego_net(edge_path, directed=True, normalize=True):
    """Reads in the ego network topology, represented as an edgelist, and generates a 
        NetworkX graph based on that edgelist. Additionally, this function will also return
        the social circles provided by the ego networks.

    Arguments:
        edge_path {str} -- Path to the edgelist file.
        circle_dir {str} -- Path to the circles *directory* where the *.circles files are located.

    Keyword Arguments:
        directed {bool} -- True if the topology is directed, False otherwise. (default: {True})
    """
    g_type = nx.DiGraph if directed else nx.Graph
    graph = nx.read_edgelist(edge_path, nodetype=int, create_using=g_type)
    if normalize:
        nodes = sorted(graph.nodes())
        mapping = {nodes[i]: i for i in range(len(nodes))}
        graph = nx.relabel.convert_node_labels_to_integers(graph)
        return graph, mapping
    else:
        return graph
	

def init_ffm(graph, rand_gen=random.random):
    """Create a dict that contains the FFM factors for each node in the graph topology.
    
    Arguments:
        graph {dict} -- Nodewise FFM factors randomly generated.
        rand_gen {function} -- Random number generator. (default: {random.random()})
    """
    ffm = {node: {'O': random.random(), 'C': random.random(), 'E': random.random(), 
                  'A': random.random(), 'N': random.random()} for node in graph.nodes()}
    return ffm
	
	
def init_opinion(graph, circles=None, rand_gen=random.random):
    """Generates a dict that provides the initial opinions of each node in `graph`. This
        function will also consider circles that are provided by the ego networks. If we
        are considering `circles` (i.e., `circles` is not None), then we assign initial
        opinions in a homophilic fashion. In that case, opinions are given to each circle
        and each node adopts the opinion of its circle or the average of the opinions of
        each circle it belongs to if it belongs to more than one circle.  

    Arguments:
        graph {nx.Graph} -- Graph under consideration.

    Keyword Arguments:
        circles {dict} -- Circles with the nodes that belong in them. (default: {None})
    """
    if circles is None:
        opinion = {node: {0: rand_gen()} for node in graph.nodes()}

    else:
        opinion = {node: [] for node in graph.nodes()}
        circle_opinions = {circle: rand_gen() for circle in circles.keys()}
        for c in circles.keys():
            for node in circles[c]:
                opinion[node].append(circle_opinions[c])
        opinion = {node: {0: np.mean(opinion[node])} if len(opinion[node]) != 0 else 
                         {0: rand_gen()} for node in opinion.keys()}
        
    return opinion

if __name__ == '__main__':
    g, mapping = get_ego_net(os.path.join('ego-nets', 'facebook_combined.txt'), directed=False)
    V = set(g.nodes())
    missing_node = 98317
    print(len(g.nodes()))
    print(missing_node in V)