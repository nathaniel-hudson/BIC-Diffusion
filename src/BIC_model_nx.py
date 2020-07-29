import Constants
import datetime
import numpy    as np
import pandas   as pd
import random   as rd
import networkx as nx

from BIC_model import BIC_Model

from scipy.stats import arcsine, uniform

class BIC_Model_nx(BIC_Model):

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

        ## Initialize the bevahioral influence vector, to avoid having to constantly 
        ## compute it for each user since it's a constant value.
        self.ffm_inf = np.array([self.behavioral_inf(user) for user in self.graph.nodes])
        self.prepared = False


    def prepare(self, seed_set=None, t_horizon=100):
        """Prepare the BIC model instance for simulation by instantiating an instance-wide
           opinion vector that's indexed by time-step.

        Parameters
        ----------
        t_horizon : int, optional
            Number of time-steps considered for a simulation, by default 100
        """
        # shape = (t_horizon+1, len(self.graph))
        self.opinion  = [
            self.init_opinion[node]
            for node in self.graph.nodes
        ]
        self.attempts = {
            user: 0 
            for user in self.graph.nodes
        }
        # self.state    = {
        #     node: self.ACTIVATED if node in seed_set else self.VULNERABLE
        #     for node in self.graph.nodes()
        # }
        self.prepared = True


    def total_opinion(self):
        return sum(self.opinion[node] for node in self.graph.nodes())


    def diffuse(self, active_set, killed_set, t, threshold=1):
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
        (active_set: set, killed_set: set)
            A tuple containing the the set of active/activated user nodes and user nodes that 
            have been killed (i.e., can no longer activate neighbors).
        """
        assert self.prepared, 'Need to run Model.prepare() before diffusing.'

        newly_killed   = set()
        newly_activted = set()
        for active_node in active_set:
            self.attempts[active_node] += 1

            neighbors = set(self.graph.neighbors(active_node)) - active_set
            for neighbor in neighbors:
                if rd.random() <= self.prop_prob(active_node, neighbor, t):
                    newly_activted.add(neighbor)

            if self.attempts[active_node] >= threshold:
                killed_set.add(active_node)
                newly_killed.add(active_node)

        for killed_node in newly_killed:
            active_set.remove(killed_node)
        for activated_node in newly_activted:
            active_set.add(activated_node)

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
        

        self.opinion = new_opinion

        return active_set, killed_set
            
            

        '''
        new_active_set = set(active_set.copy())
        activated = set()
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
                for out_neighbor in neighbors - active_set:
                    if rd.random() <= self.prop_prob(node, out_neighbor, t, use_attempts=True):
                        new_active_set.add(out_neighbor)
                        activated.add(out_neighbor)
                    else:
                        to_penalize.add(out_neighbor)

        ## Assign the (t+1) opinions for each user node.
        for node in self.graph.nodes():
            if node in activated:
                self.opinion[t+1][node] = 1.0
            elif node in to_penalize:
                self.opinion[t+1][node] = self.penalized_update(node, t)
            else:
                self.opinion[t+1][node] = self.general_update(node, t)
        '''

        ## Record all reached nodes (i.e., active/activated nodes and nodes that failed to activate).
        visited = new_active_set.union(to_penalize)

        return new_active_set, visited


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

        num = self.opinion[node]
        den = 1
        std = np.std([self.opinion[neighbor] for neighbor in neighbors])
        for nei in neighbors:
            if (self.opinion[node] - std <= self.opinion[nei]) and (self.opinion[nei] <= self.opinion[node] + std):
                num += self.opinion[nei] * (1 - abs(self.init_opinion[node] - self.opinion[nei]))
                den += (1 - abs(self.init_opinion[node] - self.opinion[nei]))

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

        num = self.opinion[node]
        den = 1
        std = np.std([self.opinion[neighbor] for neighbor in neighbors])
        for nei in neighbors:
            if self.opinion[nei] <= self.opinion[node] + std:
                num += self.opinion[nei] * (1 - abs(self.init_opinion[node] - self.opinion[nei]))
                den += (1 - abs(self.init_opinion[node] - self.opinion[nei]))

        return num / den


    def prop_prob(self, source, target, t, use_attempts=True):
        """Calculates the propagation probability for `source` to activate `target`.

        Parameters
        ----------
        source : int
            User node ID for the activated source node.
        target : int
            User node ID for the unactivated target node.
        t : int
            Current time-step during a simulation over a given time horizon.
        attempt : bool
            Use the attempt counter to anneal propagation probability if True, (default=True).

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

        if use_attempts == True:
            return 0 if den == 0 else (num/den) ** self.attempts[source]
        else:
            return 0 if den == 0 else (num/den)


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
        return self.opinion[target] * (1 - abs(self.init_opinion[source] - self.opinion[target]))


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


"""
Running a simulation using the BIC model with a LARGE online social network topology.
This will also analyze the runtime/load time of the networks.
"""
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time

    from Algorithms.Heuristics import random_solution, degree_solution
    from tqdm                  import tqdm

    data = {'k': [], 'opinion': [], '|A|': [], 'total-time': []}
    
    rd.seed(17)
    graph = nx.read_edgelist('topos/amazon-graph.txt', nodetype=int)
    ffm = {user: {factor: rd.random() for factor in 'OCEAN'} for user in graph.nodes}
    opinion = np.array([rd.random() for user in graph.nodes])
    model = BIC_Model_nx(graph, ffm, opinion)
    seed_sizes = list(range(0, 1001, 100)); seed_sizes[0] = 1

    for k in seed_sizes: 
        model.prepare()
        print(f'>> Running for seed size (k = {k}).')
        diff_start = time.time()
        active_set = set(degree_solution(model, k))
        killed_set = set()
        for timestep in tqdm(range(100)):
            active_set, killed_set = model.diffuse(active_set, killed_set, timestep)

        data['k'].append(k)
        data['opinion'].append(model.total_opinion())
        data['|A|'].append(len(active_set.union(killed_set)))
        data['total-time'].append(time.time() - diff_start)

    df = pd.DataFrame.from_dict(data)
    
    sns.barplot(x='k', y='|A|', data=df)
    plt.title('Activation Set vs. Seed Set Size')
    plt.show()
    plt.clf()

    sns.barplot(x='k', y='opinion', data=df)
    plt.title('Total Opinion vs. Seed Set Size')
    plt.show()
    plt.clf()

    sns.barplot(x='k', y='total-time', data=df)
    plt.title('Runtime vs. Seed Set Size')
    plt.show()
    plt.clf()