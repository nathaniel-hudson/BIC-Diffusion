import datetime
import numpy    as np
import pandas   as pd
import random   as rd
import networkx as nx

from scipy.stats import arcsine, uniform

# BIC-related constants. The Lambda constants for the FFM factors are based on a paper
# related to how they factor into information spreading.
LAMBDA_DEFAULT = {'O': -0.017, 'C': 0.062, 'E': 0.142, 'A': 0.073, 'N': 0.066}
GAMMA_DEFAULT = 1.0

class BIC(object):

    VULNERABLE = 0
    ACTIVATED  = 1
    EXHAUSTED  = 2

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

        # Initialize the bevahioral influence vector, to avoid having to constantly 
        # compute it for each user since it's a constant value.
        self.ffm_inf = np.array([self.behavioral_inf(user) for user in self.graph.nodes])
        self.prepared = False


    def prepare(self, threshold=1):
        """Prepare the BIC model instance for simulation by instantiating an instance-wide
           opinion vector that's indexed by time-step.

        Parameters
        ----------
        threshold : int, optional
            Threshold of maximal activation attempts allowed, by default 1
        """
        # shape = (t_horizon+1, len(self.graph))
        self.threshold = threshold
        self.opinion  = [
            self.init_opinion[node]
            for node in self.graph.nodes
        ]
        self.attempts = {
            user: 0 
            for user in self.graph.nodes
        }
        self.prepared = True


    def total_opinion(self):
        """Return the total current opinion over all nodes in the network.

        Returns
        -------
        float
            Total opinion.
        """
        return sum(self.opinion[node] for node in self.graph.nodes())


    def simulate(self, seed_set, time_horizon):
        """Run a simulation over a series of diffusion steps.

        Parameters
        ----------
        seed_set : set
            Nodes to be initially activated to begin opinion diffusion.
        time_horizon : int
            Number of maximum time-steps allowed (diffusion can end sooner due to satu-
            ration).

        Returns
        -------
        (total_opinion: float, activated_set: set, visited_set: set)
            A 3-tuple containing the total opinion by the end of the simulation, the set 
            of activated nodes, and the set
            of visited nodes.
        """
        if not self.prepared:
            self.prepare()

        active_set = seed_set.copy() if isinstance(seed_set, set) else set(seed_set)
        killed_set = set()
        visited_set = set()

        ## Perform diffusion steps for the simulation.
        for t in range(time_horizon+1):
            active_set, killed_set, visited, done = self.diffuse(active_set, killed_set, t)
            visited_set = visited_set.union(visited)
            if done:
                break

        activated_set = active_set.union(killed_set)
        self.prepared = False
        
        return self.total_opinion(), activated_set, visited_set


    def diffuse(self, active_set, killed_set, t):
        """Performs a single diffusion step to be used in a simulation.

        Parameters
        ----------
        active_set : set
            Set of activated nodes (i.e., active spreaders) to influence opinion.
        t : int
            Current time-step in the diffusion process.

        Returns
        -------
        (active_set: set, killed_set: set, visited: set, done: bool)
            A tuple containing the the set of active/activated user nodes and user nodes 
            that have been killed (i.e., 
            can no longer activate neighbors).
        """
        assert self.prepared, 'Need to run Model.prepare() before diffusing.'

        newly_killed   = set()
        newly_activated = set()
        not_activated  = set()
        for active_node in active_set:
            self.attempts[active_node] += 1

            neighbors = set(self.graph.neighbors(active_node)) - active_set
            for neighbor in neighbors:
                if rd.random() <= self.prop_prob(active_node, neighbor):
                    newly_activated.add(neighbor)
                else:
                    not_activated.add(neighbor)

            if self.attempts[active_node] >= self.threshold:
                killed_set.add(active_node)
                newly_killed.add(active_node)

        for killed_node in newly_killed:
            active_set.remove(killed_node)
            self.opinion[killed_node] = 1.0
        for activated_node in newly_activated:
            active_set.add(activated_node)
            self.opinion[activated_node] = 1.0

        new_opinion = self.opinion.copy()
        for active_node in active_set:
            new_opinion[active_node] = 1.0
            for neighbor in self.graph.neighbors(active_node):
                if neighbor in killed_set:
                    new_opinion[neighbor] = 1.0

                elif self.attempts[active_node] == 1:
                    new_opinion[neighbor] = self.penalized_update(neighbor, t)

                else:
                    new_opinion[neighbor] = self.general_update(neighbor, t)
        
        # Update opinion, determine if the diffusion process has saturated, get visited, 
        # nodes, and return.
        self.opinion = new_opinion
        done = len(active_set) == 0
        visited = not_activated.union(active_set).union(killed_set)
        return active_set, killed_set, visited, done


    def general_update(self, node, t):
        """Perform a GENERALIZED update for the given user node's opinion. This update is 
           for cases where there is no incurred penalty.

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

        num = self.opinion[node]
        den = 1
        std = np.std([self.opinion[neighbor] for neighbor in neighbors])
        for nei in neighbors:
            if (self.opinion[node] - std <= self.opinion[nei]) and \
               (self.opinion[nei] <= self.opinion[node] + std):
                num += self.opinion[nei] * \
                    (1 - abs(self.init_opinion[node] - self.opinion[nei]))
                den += (1 - abs(self.init_opinion[node] - self.opinion[nei]))

        return num / den


    def penalized_update(self, node, t):
        """Perform a PENALIZED update for the given user node's opinion. This update is 
           for cases where there is no incurred penalty.

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

        num = self.opinion[node]
        den = 1
        std = np.std([self.opinion[neighbor] for neighbor in neighbors])
        for nei in neighbors:
            if self.opinion[nei] <= self.opinion[node] + std:
                num += self.opinion[nei] * \
                    (1 - abs(self.init_opinion[node] - self.opinion[nei]))
                den += (1 - abs(self.init_opinion[node] - self.opinion[nei]))

        return num / den


    def prop_prob(self, source, target, use_attempts=True):
        """Calculates the propagation probability for `source` to activate `target`.

        Parameters
        ----------
        source : int
            User node ID for the activated source node.
        target : int
            User node ID for the unactivated target node.
        t : int
            Current time-step during a simulation over a given time horizon.
        attempt : bool, optional
            Use the attempt counter to anneal propagation probability if True, 
            (default=True).

        Returns
        -------
        float
            Propagation probability for `source` to activate `target`.
        """
        if not self.graph.has_edge(source, target):
            return 0

        in_neighbors = self.graph.neighbors(target) \
                       if not nx.is_directed(self.graph) \
                       else self.graph.predecessors(target)

        opinion_inf = self.opinion_inf(source, target)
        behavio_inf = self.ffm_inf[target]

        num = behavio_inf * opinion_inf
        den = sum([self.opinion_inf(in_neighbor, target)
                   for in_neighbor in in_neighbors])

        if use_attempts == True:
            return 0 if den == 0 else (num/den) ** self.attempts[source]
        else:
            return 0 if den == 0 else (num/den)


    def opinion_inf(self, source, target):
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
        # In some cases, the `prop_prob()` function might be called without `prepare()`
        # being called. As a result, references to `self.opinion` can be invalid. Thus,
        # we need to do this check first.
        if self.prepared:
            return self.opinion[target] * \
                (1 - abs(self.init_opinion[source] - self.opinion[target]))
        else:
            return self.init_opinion[target] * \
                (1 - abs(self.init_opinion[source] - self.init_opinion[target]))


    def behavioral_inf(self, user, coeffs=LAMBDA_DEFAULT):
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

        if (old_min != old_max) and (new_min != new_max):
            influence = (((beta - old_min) * (new_max - new_min)) /
                        (old_max - old_min)) + new_min
        else:
            influence = (new_max + new_min) / 2

        return influence


"""
Running a simulation using the BIC model with a LARGE online social network topology.
This will also analyze the runtime/load time of the networks.
"""
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3, 4, 5, 6, 7])
    graph.add_edges_from([
        (2, 1), (2, 3), (2, 4),
        (3, 5),
        (4, 5), (4, 6), (4, 7),
        (6, 7)
    ])
    