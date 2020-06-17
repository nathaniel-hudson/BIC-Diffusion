import Constants
import datetime
import networkx as nx
import numpy    as np
import pandas   as pd
import random   as rd

from scipy.stats import arcsine, uniform

class BIC_Model:

    def __init__(self, graph, ffm, init_opinion):
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
        self.graph   = graph   # Social network topology.
        self.ffm     = ffm     # Behavioral (FFM) parameters.
        self.init_opinion = init_opinion  # Initial opinion vector.

        ## Initialize the bevahioral influence vector, to avoid having to constantly 
        ## compute it for each user since it's a constant value.
        self.ffm_inf = np.array([self.behavioral_inf(user) for user in self.graph.nodes])


    def prepare(self, t_horizon=100):
        """Prepare the BIC model instance for simulation by instantiating an instance-wide
           opinion vector that's indexed by time-step.

        Parameters
        ----------
        t_horizon : int, optional
            Number of time-steps considered for a simulation, by default 100
        """
        shape = (t_horizon, len(self.graph))
        self.opinion = np.append(self.init_opinion, np.zeros(shape), axis=0)


    def diffuse(self, active_set, attempts, t, threshold=1):
        """Performs a single diffusion step to be used in a simulation.

        Parameters
        ----------
        active_set : set
            Set of activated nodes (i.e., active spreaders) to influence opinion.
        attempts : dict
            Data structure tracking the number of attempts of each pair of nodes.
        t : int
            Current time-step in the diffusion process.
        threshold : int, optional
            Threshold of maximal activation attempts allowed, by default 1

        Returns
        -------
        set
            Set of newly activated nodes.
        """
        new_active_set = set(active_set.copy())
        to_penalize = set()

        ## Iterate through each activated node in `active_set` to perform activation.
        for node in set(active_set):
            ## Grab the out-neighbors for the currently considered activated node and
            ## iterate its attempts tracker.
            neighbors = set(graph.neighbors(node))
            attempts[node] += 1

            ## If the current active node has not exceeded the threshold, attempt to
            ## activate its unactived out-neighbors.
            if attempts[node] <= threshold:
                for out_neighbor in neighbors - new_active_set:
                    if rd.random() <= self.prop_prob(node, out_neighbor, attempts[node], t):
                        new_active_set.add(out_neighbor)
                    else:
                        to_penalize.add(out_neighbor)

        for node in graph.nodes():
            if node in new_active_set:
                opinion[node][t+1] = 1.0
            elif node in to_penalize:
                opinion[node][t+1] = penalized_update(graph, node, opinion, t)
            else:
                opinion[node][t+1] = general_update(graph, node, opinion, t)

        return new_active_set



    def prop_prob(self, source, target, attempt, t):
        """Calculates the propagation probability for `source` to activate `target`.

        Parameters
        ----------
        source : int
            User node ID for the activated source node.
        target : int
            User node ID for the unactivated target node.
        attempt : int
            The current activation attempt.
        t : int
            Current time-step during a simulation over a given time horizon.

        Returns
        -------
        float
            Propagation probability for `source` to activate `target`.
        """
        in_neighbors = graph.neighbors(target) \
                       if not nx.is_directed(self.graph) \
                       else graph.predecessors(target)

        opinion_inf = self.opinion_inf(source, target, t)
        behavio_inf = self.ffm_inf[target]

        num = behavio_inf * opinion_inf
        den = sum([self.opinion_inf(in_neighbor, target, t)
                   for in_neighbor in in_neighbors])

        return 0 if den == 0 else (num/den) ** attempt


    def opinion_inf(self, source, target, t):
        return self.opinion[target][t] * (1 - abs(self.opinion[source][0] - self.opinion[target][t]))


    def behavioral_inf(self, user, coeffs=Constants.LAMBDA_DEFAULT):
        beta_pos = sum(coeffs[factor] * self.ffm[user][factor]
                       for factor in coeffs if coeffs[factor] >= 0)
        beta_neg = sum(coeffs[factor] * self.ffm[user][factor]
                       for factor in coeffs if coeffs[factor] < 0)

        beta = beta_pos + beta_neg

        old_max = sum(coeffs[factor]
                      for factor in coeffs if coeffs[factor] >= 0)
        old_min = sum(coeffs[factor]
                      for factor in coeffs if coeffs[factor] < 0)
        new_max = 1.0
        new_min = 0.0

        if old_min != old_max and new_min != new_max:
            influence = (((beta - old_min) * (new_max - new_min)) /
                        (old_max - old_min)) + new_min
        else:
            influence = (new_max + new_min) / 2

        return influence


