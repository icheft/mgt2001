# samp for sampling
import pandas as pd
import math
import numpy as np
import scipy.stats as stats
import itertools
import random


def to_xbar_freq(possible_outcome, repeat=1):
    """
    Input: possible_outcome 
    Return a `DataFrame` that stores `x` values and its `prob` (probability).
    """
    xbar_all = []
    for aelem in itertools.product(possible_outcome, repeat=repeat):
        xbar_all.append(np.mean(aelem))

    xbar_df = pd.DataFrame({'xbar': xbar_all})
    xbar_freq = xbar_df.xbar.value_counts()
    xbar_freq = xbar_freq.to_frame().reset_index()
    xbar_freq = xbar_freq.sort_values(by="index")
    xbar_freq = xbar_freq.rename(columns={'index': 'x', 'xbar': 'freq'})
    tmp1 = xbar_freq.freq.sum()
    xbar_freq['prob'] = xbar_freq.freq / tmp1
    return xbar_freq


def sumprob(xfreq, bm=3, tp=4):
    """
    Input: a list of frequency, bottom limit, top limit
    Output: sum
    """
    tmpind1 = xfreq.x >= bm
    tmpind2 = xfreq.x <= tp
    return xfreq[tmpind1 & tmpind2].prob.sum()


def returnZ(x, mu, std):
    """
Usage for sampling distribution of the difference between two means:

z = mgt2001.samp.returnZ(x, mgt2001.samp.returnE(mu1, mu2), mgt2001.samp.returnStd(std1, std2, n1, n2))
    """
    return (x - mu) / std


def check5(n, p):
    """
    Return a bool. Check if a set of measurements can apply normal approximation.
    """
    return n * p > 5 and n * (1 - p) > 5


def returnE(x1, x2):
    return x1 - x2


def returnStd(std1, std2, n1, n2):
    return math.sqrt((std1 ** 2) / n1 + (std2 ** 2) / n2)


def returnVar(std1, std2, n1, n2):
    return math.pow(returnStd(std1, std2, n1, n2), 2)


# Estimation


def con_level(x_bar, sigma, n, alpha, show=True):
    """
Input: x_bar (x_mean), sigma, sample size, alpha, show=True
Return the confidence level at $\alpha$. Return a dictionary: {"lcl": lcl, "ucl": ucl, "x_bar": x_bar, "z_value": z_value, "sig_x_bar": sig_x_bar}

+ `show`: default is `True`. Set to `False` to disable rendering.
    """
    a = alpha
    con_coef = 1 - a
    z_value = stats.norm.ppf(1 - a / 2)
    sig_x_bar = sigma / math.sqrt(n)
    lcl = x_bar - z_value * sig_x_bar
    ucl = x_bar + z_value * sig_x_bar
    result = f"""{con_coef * 100:.1f}% Confidence Interval: [{lcl:.4f}, {ucl:.4f}]
Mean: {x_bar:.4f}
Sample Size: {n}
Z-Value: {z_value:.4f}
Sigma of X-Bar: {sig_x_bar:.4f}
    """
    if show:
        print(result)
    return {"lcl": lcl, "ucl": ucl, "x_bar": x_bar, "z_value": z_value, "sig_x_bar": sig_x_bar}


def bound(a, std, n):
    """
    Calculate the bound.

    Input: significance level, sigma, sample size
    Output: bound
    """
    z_value = stats.norm.ppf(1 - a / 2)
    return z_value * std / math.sqrt(n)


def estimate_samp_size(bound, sig_level, sigma):
    """
    Input: bound, significance level, sigma
    Output: the estimated sample size n (not rounded)
    """
    a = sig_level
    z_value = stats.norm.ppf(1 - a / 2)
    n = math.pow(z_value * sigma / bound, 2)
    return n
