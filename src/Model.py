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
        self.graph = graph   # Social network topology.
        self.ffm   = ffm     # Behavioral (FFM) parameters.
        self.init_opinion = init_opinion  # Initial opinion vector.

        ## Initialize the bevahioral influence vector, to avoid having to constantly 
        ## compute it for each user since it's a constant value.
        self.ffm_inf = np.array([self.behavioral_inf(user) for user in self.graph.nodes])
        self.prepared = False


    def prepare(self, t_horizon=100):
        """Prepare the BIC model instance for simulation by instantiating an instance-wide
           opinion vector that's indexed by time-step.

        Parameters
        ----------
        t_horizon : int, optional
            Number of time-steps considered for a simulation, by default 100
        """
        shape = (t_horizon, len(self.graph))
        self.opinion = np.append([self.init_opinion], np.zeros(shape), axis=0)
        self.attempts = {user: 0 for user in self.graph.nodes}
        self.prepared = True


    def diffuse(self, active_set, t, threshold=1):
        """Performs a single diffusion step to be used in a simulation.

        Parameters
        ----------
        active_set : set
            Set of activated nodes (i.e., active spreaders) to influence opinion.
        t : int
            Current time-step in the diffusion process.
        threshold : int, optional
            Threshold of maximal activation attempts allowed, by default 1

        Returns
        -------
        set
            Set of newly activated nodes.
        """
        assert self.prepared, 'Need to run Model.prepare() before diffusing.'

        new_active_set = set(active_set.copy())
        to_penalize = set()

        ## Iterate through each activated node in `active_set` to perform activation.
        for node in set(active_set):
            ## Grab the out-neighbors for the currently considered activated node and
            ## iterate its attempts tracker.
            neighbors = set(self.graph.neighbors(node))
            self.attempts[node] += 1

            ## If the current active node has not exceeded the threshold, attempt to
            ## activate its unactived out-neighbors.
            if self.attempts[node] <= threshold:
                for out_neighbor in neighbors - new_active_set:
                    if rd.random() <= self.prop_prob(node, out_neighbor, self.attempts[node], t):
                        new_active_set.add(out_neighbor)
                    else:
                        to_penalize.add(out_neighbor)

        ## Assign the (t+1) opinions for each user node.
        for node in self.graph.nodes():
            if node in new_active_set:
                self.opinion[t+1][node] = 1.0
            elif node in to_penalize:
                self.opinion[t+1][node] = self.penalized_update(node, t)
            else:
                self.opinion[t+1][node] = self.general_update(node, t)

        return new_active_set


    def general_update(self, node, t):
        """Perform a GENERALIZED update for the given user node's opinion. This update is for cases where there is no incurred penalty.

        Parameters
        ----------
        node : int
            User node ID.
        t : int
            Current time-step.

        Returns
        -------
        float
            User node's updated opinion for the (t+1) time-step.
        """
        neighbors = self.graph.neighbors(node) \
                    if not nx.is_directed(self.graph) \
                    else self.graph.predecessors(node)

        # num = self.opinion[node][0]
        num = self.opinion[t][node]
        den = 1
        std = np.std([self.opinion[t][neighbor] for neighbor in neighbors])
        for nei in neighbors:
            if (self.opinion[t][node] - std <= self.opinion[t][nei]) and (self.opinion[t][nei] <= self.opinion[t][node] + std):
                num += self.opinion[t][nei] * (1 - abs(self.opinion[0][node] - self.opinion[t][nei]))
                den += (1 - abs(self.opinion[0][node] - self.opinion[t][nei]))

        return num / den


    def penalized_update(self, node, t):
        """Perform a PENALIZED update for the given user node's opinion. This update is for cases where there is no incurred penalty.

        Parameters
        ----------
        node : int
            User node ID.
        t : int
            Current time-step.

        Returns
        -------
        float
            User node's penalized, updated opinion for the (t+1) time-step.
        """
        neighbors = self.graph.neighbors(node) \
                    if not nx.is_directed(self.graph) \
                    else self.graph.predecessors(node)

        # num = self.opinion[node][0]
        num = self.opinion[t][node]
        den = 1
        std = np.std([self.opinion[t][neighbor] for neighbor in neighbors])
        for nei in neighbors:
            if self.opinion[t][nei] <= self.opinion[t][node] + std:
                num += self.opinion[t][nei] * (1 - abs(self.opinion[0][node] - self.opinion[t][nei]))
                den += (1 - abs(self.opinion[0][node] - self.opinion[t][nei]))

        return num / den


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
        in_neighbors = self.graph.neighbors(target) \
                       if not nx.is_directed(self.graph) \
                       else self.graph.predecessors(target)

        opinion_inf = self.opinion_inf(source, target, t)
        behavio_inf = self.ffm_inf[target]

        num = behavio_inf * opinion_inf
        den = sum([self.opinion_inf(in_neighbor, target, t)
                   for in_neighbor in in_neighbors])

        return 0 if den == 0 else (num/den) ** attempt


    def opinion_inf(self, source, target, t):
        """Calculates the impact of opinion on propagation probabilities.

        Parameters
        ----------
        source : int
            User node ID of activated source node.
        target : int
            User node ID of targeted node.
        t : int
            Current time-step.

        Returns
        -------
        float
            Impact of opinion on propagation probability.
        """
        return self.opinion[t][target] * (1 - abs(self.opinion[0][source] - self.opinion[t][target]))


    def behavioral_inf(self, user, coeffs=Constants.LAMBDA_DEFAULT):
        """Calculates the impact of FFM factors (behavior) on propagation probabilities.

        Parameters
        ----------
        user : int
            User node ID.
        coeffs : dict, optional
            Behavioral coefficients for FFM factors, by default Constants.LAMBDA_DEFAULT

        Returns
        -------
        float
            Impact of behavior on propagation probability.
        """
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


