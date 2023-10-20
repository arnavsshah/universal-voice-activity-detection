import itertools
from typing import Iterable

def pairwise(iterable: Iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def merge_dict(defaults: dict, custom: dict = None):
    """merge 2 dictionaries, with the custom dict values overriding the default dict values"""
    params = dict(defaults)
    if custom is not None:
        params.update(custom)
    return params