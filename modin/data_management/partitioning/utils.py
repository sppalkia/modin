from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def compute_chunksize(length, num_splits):
    # We do this to avoid zeros and having an extremely large last partition
    return length // num_splits if length % num_splits == 0 \
        else length // num_splits + 1
