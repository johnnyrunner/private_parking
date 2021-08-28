import random
from functools import wraps
from time import time

import numpy as np

REAL_COUNTS_LIST = 'real_counts_list'

ESTIMATED_COUNTS_LIST = 'estimated_counts_list'

CHANGES = 'changes_'

def timing(f):
    # a decorator for function timinf
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # print('func:%r args:[%r, %r] took: %2.4f sec' % \
        #       (f.__name__, args, kw, te - ts))
        print('func:%r took: %2.4f sec' % \
              (f.__name__, te - ts))
        return result

    return wrap


def set_random_seed(random_seed):
    # set a random seed for both random and numpy
    random.seed(random_seed)
    np.random.seed(random_seed)