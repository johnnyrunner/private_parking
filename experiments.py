import copy
import itertools
import multiprocessing
import time
import numpy as np

import pandas as pd

import tqdm

from evaluation import mean_relative_error
from personalized_count_estimator import pce_runner
from simulation import simulate_static_situation, change_static_simulation, make_dynamic_effects_dictionary
from utils import ESTIMATED_COUNTS_LIST, REAL_COUNTS_LIST, CHANGES, set_random_seed, timing


def static_situation_investigate():
    '''
    A one-core static evaluation of different - run this function for Experiment #1
    :return: save a file to the appropriate folder of the results of the experiment - for all the different params
    '''
    results_dictionaries = []
    for epsilons in [[0.25, 0.5, 0.75], [0.75, 1.0, 1.25]]:
        for num_users in [1000, 10000, 100000]:
            for height in [1, 2, 3]:
                inputs_dictionary, users = simulate_static_situation(epsilons=epsilons, num_users=num_users,
                                                                     height=height)
                results_dictionary = pce_runner(users, **inputs_dictionary)
                results_dictionary.update(inputs_dictionary)
                results_dictionaries.append(results_dictionary)
    df = pd.DataFrame(results_dictionaries)
    df = df.drop('taxonomy', axis=1)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    df.to_excel(f"results/static_situation/{timestr}.xlsx")


def get_chunks(iterable, chunks=1):
    # This is from http://stackoverflow.com/a/2136090/2073595
    lst = list(iterable)
    return [lst[i::chunks] for i in range(chunks)]


def dynamic_experiment_worker(pairs):
    '''
    helper function for running the dynamic experiment  (Experiment #2)
    This function actually outputs the results of Experiment #1 too.

    It is making a users list
    check the ability of the server-pce on it
    adds/delete users form the list
    check the ability of the server-pce on it, again

    :param pairs:
    :return: results dictionaries for the duull dynamic experiment
    '''
    results_dictionaries = []

    for epsilons, num_users, height, deletion_probability, addition_probability, random_seed in pairs:
        set_random_seed(random_seed)
        inputs_dictionary, users = simulate_static_situation(epsilons=epsilons, num_users=num_users,
                                                             height=height)
        original_results_dictionary = pce_runner(users, **inputs_dictionary)
        inputs_dictionary_change, changed_users = change_static_simulation(users, deletion_probability,
                                                                           addition_probability,
                                                                           inputs_dictionary)

        changed_results_dictionary = pce_runner(changed_users, **inputs_dictionary)
        dynamic_effects_dictionary = make_dynamic_effects_dictionary(original_results_dictionary,
                                                                     changed_results_dictionary)

        mean_relative_dynamic_change, changes_list = mean_relative_error(
            dynamic_effects_dictionary[CHANGES + ESTIMATED_COUNTS_LIST],
            dynamic_effects_dictionary[CHANGES + REAL_COUNTS_LIST],
            sanity_bound=0.0001)

        original_results_dictionary['changes_list'] = changes_list
        original_results_dictionary['random_seed'] = random_seed
        original_results_dictionary.update(dynamic_effects_dictionary)
        original_results_dictionary.update(inputs_dictionary)
        original_results_dictionary['mean_relative_dynamic_change'] = mean_relative_dynamic_change
        original_results_dictionary['deletion_probability'] = deletion_probability
        original_results_dictionary['addition_probability'] = addition_probability
        results_dictionaries.append(original_results_dictionary)

    return results_dictionaries


@timing
def dynamic_situation_investigate():
    '''
    This function actually runs the dynamic experiment with the hyperparameters I reported in the handout.
    :return:
    '''
    epsilons_lists = [[0.25, 0.5, 0.75]] #[[0.75, 1.0, 1.25]]  # ,
    num_users_lists = [10000, 50000, 100000, 300000]
    heights_lists = [1, 2, 3]
    deletion_probability_list = [0.01, 0.5]
    addition_probability_list = [0.01, 0.5]
    random_seed = list(range(5))
    products = itertools.product(epsilons_lists, num_users_lists, heights_lists, deletion_probability_list,
                                 addition_probability_list, random_seed)
    products_copy = copy.deepcopy(products)
    print(f'number of products is: {len(list(products_copy))}')
    chunked_pairs = get_chunks(products, chunks=multiprocessing.cpu_count())
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        results_dictionaries_list_lists = tqdm.tqdm(p.map(dynamic_experiment_worker, chunked_pairs))
    results_dictionaries_lists = [item for sublist in results_dictionaries_list_lists for item in sublist]
    df = pd.DataFrame(results_dictionaries_lists)
    df = df.groupby(by=['num_users', 'height', 'beta', 'deletion_probability', 'addition_probability']).agg(
        {'mean_relative_dynamic_change': [np.mean, np.std], 'kl_divergence_value': [np.mean, np.std],
         'mean_relative_error_value': [np.mean, np.std]})
    df = df.sort_values(by=('mean_relative_dynamic_change', 'mean'))
    timestr = time.strftime("%Y%m%d-%H%M%S")
    df.to_excel(f"results/dynamic_situation/{timestr}_{epsilons_lists}.xlsx")


if __name__ == '__main__':
    # run this file!
    # static_situation_investigate()
    dynamic_situation_investigate()
