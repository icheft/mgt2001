import numpy as np
import math as math


def percentile(data, percentage):
    """
    Will return the value corresponding to the percentile from the data.

    Difference from numpy.percentile: this function will perform a (n + 1) instead of (n) only
    """
    if type(data) == np.ndarray:
        all_data = data.copy()
        data = data.copy()
    else:
        all_data = data.values.copy()
        data = data.values.copy()
    all_data.sort()
    n = all_data.size
    l = (n + 1) * percentage / 100 - 1

    f_l = math.floor(l)
    c_l = math.ceil(l)

    percentile_val = all_data[f_l] + \
        (all_data[c_l] - all_data[f_l]) * (l - f_l)

    return percentile_val


def percentrank(data, value):
    """
    Will return the corresponding percentile rank given the value.
    """
    if type(data) == np.ndarray:
        all_data = data.copy()
        data = data.copy()
    else:
        all_data = data.values.copy()
        data = data.values.copy()
    all_data.sort()
    l = np.searchsorted(all_data, value, side='left') + 1
    n = all_data.size
    percentage = l * 100 / (n + 1)
    return percentage
