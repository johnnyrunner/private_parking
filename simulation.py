import copy

import numpy as np
from typing import Dict, List

from locations_taxonomy import LocationsTaxonomy, TelAviv_json_dir
from user import SystemUser
from utils import REAL_COUNTS_LIST, ESTIMATED_COUNTS_LIST, CHANGES


def simulate_static_situation(taxonomy=LocationsTaxonomy(TelAviv_json_dir),
                              epsilons=[1], num_users=100, beta=0.1, height=1, **args):
    '''
    build a static situation of people parking in the city - using the
    changing the seed affects it as the get_random_static_user is affected by it.

    :param taxonomy:  the taxonomy of the system
    :param epsilons: the epsilons each user is randomly choosing from
    :param num_users: number of users the static situation should conjtain
    :param beta: the accuracy param
    :param height: the height each user is Ok with the server seeing
    :param args: just to being able to call the function using a dictionary of params
    :return: inputs of the function - for documentation of experiments, users list - for usage
    '''
    inputs_dictionary = {
        'epsilons': epsilons,
        'num_users': num_users,
        'height': height,
        'beta': beta,
        'taxonomy_hash': hash(taxonomy), #irrelevant when changin the seed
        'taxonomy': taxonomy

    }
    users = [SystemUser.get_random_static_user(taxonomy, epsilons, height=height) for i in range(num_users)]
    return inputs_dictionary, users


def change_static_simulation(users: List[SystemUser], deletion_probability: float, addition_probability: float, inputs_dict):
    '''
    change a static simulation so it will contain the right users after addition and deletion to dynamic simulaton
    :param users:
    :param deletion_probability: the probability a user will be deleted
    :param addition_probability: portion of users to add to the situation
    :param inputs_dict: the doictionary to make new static situation with the *new* users only
    :return: the inputs dictionary for documentation, and the users that were added
    '''
    inputs_dict = copy.deepcopy(inputs_dict)
    inputs_dict['num_users'] = int(addition_probability * len(users))
    stayed_users = np.random.choice(users, int(len(users) * (1 - deletion_probability)))
    inputs_dictionary, more_users = simulate_static_situation(**inputs_dict)
    new_users = list(stayed_users) + list(more_users)
    return inputs_dictionary, new_users


def make_dynamic_effects_dictionary(original_results: Dict, dynamically_changed_results: Dict):
    '''
    
    :param original_results: the results of the pce on users list before changes
    :param dynamically_changed_results: the results of the pce on  users list after changed
    :return: a dictionary of the changes - what we compare in the dynamic experiment
    '''
    new_results = {}
    new_results[CHANGES + REAL_COUNTS_LIST] = [i - j for i, j in zip(original_results[REAL_COUNTS_LIST],
                                                                     dynamically_changed_results[REAL_COUNTS_LIST])]
    new_results[CHANGES + ESTIMATED_COUNTS_LIST] = [i - j for i, j in zip(original_results[ESTIMATED_COUNTS_LIST],
                                                                          dynamically_changed_results[ESTIMATED_COUNTS_LIST])]
    return new_results


