import Constants
import datetime
import numpy    as np
import pandas   as pd
import random   as rd
import networkx as nx

from igraph import * 

from scipy.stats import arcsine, uniform

class BIC_Model(object):
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
        raise Exception('Cannot initialize instance of abstract class `BIC_Model`.')


    def prepare(self, t_horizon=100):
        """Prepare the BIC model instance for simulation by instantiating an instance-wide
           opinion vector that's indexed by time-step.

        Parameters
        ----------
        t_horizon : int, optional
            Number of time-steps considered for a simulation, by default 100
        """
        raise Exception('Cannot run class method for abstract class `BIC_Model`.')


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
        (active_set: set, visited: set)
            A tuple containing the the set of active/activated user nodes and user nodes that have been visited in total.
        """
        raise Exception('Cannot run class method for abstract class `BIC_Model`.')


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
        raise Exception('Cannot run class method for abstract class `BIC_Model`.')


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
        raise Exception('Cannot run class method for abstract class `BIC_Model`.')


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
        attempt : int
            The current activation attempt.

        Returns
        -------
        float
            Propagation probability for `source` to activate `target`.
        """
        raise Exception('Cannot run class method for abstract class `BIC_Model`.')


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
        raise Exception('Cannot run class method for abstract class `BIC_Model`.')


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
        raise Exception('Cannot run class method for abstract class `BIC_Model`.')


