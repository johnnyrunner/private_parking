import math
import random
from typing import List

import numpy as np

from evaluation import evaluate_results_dictionary
from locations_taxonomy import LocationsTaxonomy
from simulation import simulate_static_situation
from user import SystemUser
from utils import timing

SMALL_NUMBER = 0.0001


def basic_clusteting(users_list: List[SystemUser]):
    '''
    users are clustered by their tau - the place they are ok with showing to the system.
    :param users_list: the users to cluster
    :return: a dictionary - for each tau - the users under it
    '''
    users_clustering = dict()
    for user in users_list:
        if user.tau in users_clustering:
            users_clustering[user.tau].append(user)
        else:
            users_clustering[user.tau] = [user]
    return users_clustering

def server_pce(beta, users: List[SystemUser], taxonomy: LocationsTaxonomy):
    '''
    This is the function that runs it all - the server:
    1. get all the taus of all the users in the clusterand make sure they are the same
    2. converts between the ids of the leaves in the subtree and the ids of the nodes
    3. calculates everything needed for the algorithm
    4. for each user - it asks the user itself (as in user.local_randomizer() for its data) - (this happens in the client side)
    5. translate all the counts back to the current tau's size

    :param beta: the accuracy parameter
    :param users: all the users in the cluster.
    :param taxonomy: the taxonomy we build on.
    :return: count estimation fot tau, as all its users are in this cluster.
    '''
    taus = [user.tau for user in users]
    taus_indices = [t.index for t in taus]
    assert len(np.unique(taus_indices)) == 1
    tau = taus[0]
    tau_leaves_num = taxonomy.get_number_leaves(tau)

    node_ids = [i for (i, _) in taxonomy.get_leaves_enumerated(tau)]
    node_ids = list(sorted(node_ids))
    node_id_to_leaf_id = {node_ids[index]: index for index in range(len(node_ids))}

    number_of_users = len(users)

    delta = math.sqrt(math.log(2 * tau_leaves_num / beta) / number_of_users)
    m = int(math.log(tau_leaves_num + 1) * math.log(2 / beta) / (delta ** 2))
    phi = np.random.choice([-1 / math.sqrt(m), 1 / math.sqrt(m)], size=(m, tau_leaves_num), p=[0.5, 0.5])
    m_list = [i for i in range(m)]
    z = np.zeros((m, 1))

    for user in users:
        j = random.choice(m_list)
        phi_jth_row = phi[j, :]
        zi = user.local_randomizer(phi_jth_row, node_id_to_leaf_id, m)
        z[j, 0] += zi

    f = {key: 0 for key in node_id_to_leaf_id.keys()}
    for i, leaf in taxonomy.get_leaves_enumerated(tau):
        f[i] = int(np.dot(z.T, phi[:, node_id_to_leaf_id[i]]))
    return f

@timing
def pce_runner(users, taxonomy, beta, **args):
    '''
    This is the runner of the experiment inside the server
    it is taking all the users and clustering only by their tau (that is achievacle by the server)
    - this happens ever few minute over the day, with all the users that contacted the server till that minute, and
    is compared with the state of the night, or a few minutes before (as done in experiment #2)
    :param users: the users to ask for taus
    :param taxonomy: the taxonomy of the server now
    :param beta: te accuracy param
    :param args: other params, for easy usage of the function
    :return: the evaluation of the results of the counts vs. the fincal counts
    '''
    users_clustering_by_taus = basic_clusteting(users)
    final_counts = {}
    final_real_counts = {}
    for tau in users_clustering_by_taus.keys():
        indices_nodes_tau_leaves = taxonomy.get_leaves_enumerated(tau)
        real_f = dict()
        for index, node in indices_nodes_tau_leaves:
            real_f[index] = 0
        for user in users_clustering_by_taus[tau]:
            if user.location_index in real_f:
                real_f[user.location_index] += 1
            else:
                assert False
        f = server_pce(beta, users_clustering_by_taus[tau], taxonomy)
        final_counts.update(f)
        final_real_counts.update(real_f)

    results_dictionary = evaluate_results_dictionary(final_real_counts, final_counts)
    return results_dictionary

if __name__ == '__main__':
    inputs_dictionary, users = simulate_static_situation()
    pce_runner(users, **inputs_dictionary)

