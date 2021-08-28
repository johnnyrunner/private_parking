import math
from typing import List

import numpy as np

from utils import ESTIMATED_COUNTS_LIST, REAL_COUNTS_LIST

SMALL_NUMBER = 0.01


def _normalize(q):
    return np.array(q) / np.linalg.norm(q)


def _positivize(q):
    return np.maximum(np.array(q), SMALL_NUMBER)


def kl_divergence(estimated_counts, real_counts):
    '''
    KL-divergence of the estimated vs. real counts, actually not used
    :param estimated_counts: a list of the estimated counts by our method
    :param real_counts: a list of the real conts of the method
    :return:
    '''
    estimated_counts = _normalize(_positivize(estimated_counts))
    assert (np.array(real_counts) >= 0).all()
    real_counts = _normalize(_positivize(real_counts))
    inside_kl = estimated_counts * np.log(estimated_counts / real_counts)
    where_kl = np.where(estimated_counts != 0, inside_kl, 0)
    return np.sum(where_kl)


def mean_relative_error(estimated_counts: List[int], real_counts: List[int], sanity_bound: int):
    '''
    a calculation of the MRE as defined in the paper
    :param estimated_counts:
    :param real_counts:
    :param sanity_bound:
    :return:
    '''
    assert len(estimated_counts) == len(real_counts)
    changes_list = [math.fabs(math.fabs(real) - math.fabs(estimated)) / max(math.fabs(real), sanity_bound) for real, estimated in
                    zip(estimated_counts, real_counts)]
    return np.mean(changes_list), changes_list


def evaluate_results_dictionary(real_counts, estimated_private_counts, sanity_bound=1):
    '''
    print and return evaluation of the estimated and real counts
    :param real_counts:
    :param estimated_private_counts:
    :param sanity_bound:
    :return:
    '''
    assert real_counts.keys() == estimated_private_counts.keys()
    real_counts_list = []
    estimated_counts_list = []
    for key, value in real_counts.items():
        real_counts_list.append(value)
        estimated_counts_list.append(estimated_private_counts[key])

    kl_divergence_value = kl_divergence(estimated_counts_list, real_counts_list)
    print('----------------------------')
    print(f'kl-divergence: {kl_divergence_value}')
    relative_error_value, _ = mean_relative_error(estimated_counts_list, real_counts_list, sanity_bound=sanity_bound)
    # print(f'sanity bound relative error: {sanity_bound}')
    print(f'mean relative error: {relative_error_value}')
    print('----------------------------')
    return {'kl_divergence_value': kl_divergence_value, 'mean_relative_error_value': relative_error_value,
            ESTIMATED_COUNTS_LIST: estimated_counts_list, REAL_COUNTS_LIST: real_counts_list}
