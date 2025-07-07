import numpy as np


def normalize(a, q=(0, 1)):
    amax, amin = np.nanquantile(a, q[1]), np.nanquantile(a, q[0])
    diff = amax - amin
    if diff > 0:
        return ((a - amin) / diff).clip(0, 1)
    else:
        return a


def compute_corr(a, b):
    a, b = np.nan_to_num(a).reshape(-1), np.nan_to_num(b).reshape(-1)
    return np.corrcoef(a, b)[0, 1]


def split_path(p):
    plst = p.split("/")
    return "/".join(plst[:-1]), plst[-1]
