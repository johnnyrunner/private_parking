import math
import random

import numpy as np

SMALL_NUMBER = 0.0001


class SystemUser:
    '''
    this is the representing a user in the system
    init one for a new user, "get_random_static_user" for user for a certain system
    '''
    def __init__(self, epsilon=1, tau=None, location_node=None, location_index=None):
        self.epsilon = epsilon
        assert tau is not None
        self.tau = tau
        assert location_node is not None
        self.location_node = location_node
        assert location_index is not None
        self.location_index = location_index

    def local_randomizer(self, x: np.array, node_id_to_leaf_id, m):
        '''
        the user's local randomizer - defining what to send back to the server
        :param x: the vector recieved from the server
        :param location_index: the index of the location of the specific user
        :param epsilon: the epsilon of the specific user
        :param m: size of the space
        :return:
        '''
        x_li = x[node_id_to_leaf_id[self.location_index]]
        return _get_zi(self.epsilon, m, x_li)

    @staticmethod
    def get_random_static_user(taxonomy, epsilons, height):
        '''
        
        :param taxonomy: the taxonomy of the system the user participates int
        :param epsilons: privacy epsilon of the user
        :param height: the height that the user is ok with showing the server
        :return: an appropriate User
        '''
        epsilon = random.choice(epsilons)
        tau = taxonomy.get_random_node(height=height)
        location_index, location_node = taxonomy.get_random_leaf(tau)
        return SystemUser(epsilon, tau, location_node, location_index)


def _calc_c_epsilon(epsilon):
    exp_eps = math.exp(epsilon)
    c_eps = (exp_eps + 1) / (exp_eps - 1)
    return c_eps, exp_eps


def _get_zi(epsilon, m, x_li):
    c_eps, exp_eps = _calc_c_epsilon(epsilon)
    prob_plus = exp_eps / (exp_eps + 1)
    prob_minus = 1 / (exp_eps + 1)
    assert math.fabs(prob_plus + prob_minus - 1) < SMALL_NUMBER
    sign = np.random.choice([1, -1], size=1, p=[prob_plus, prob_minus])
    return sign * c_eps * m * x_li


