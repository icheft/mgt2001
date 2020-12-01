# samp for sampling
import pandas as pd
import math


def to_xbar_freq(possible_outcome, repeat=1):
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
    tmpind1 = xfreq.x >= bm
    tmpind2 = xfreq.x <= tp
    return xfreq[tmpind1 & tmpind2].prob.sum()


def returnZ(x, mu, std):
    return (x - mu) / std


def check5(n, p):
    return n * p > 5 and n * (1 - p) > 5


def returnE(x1, x2):
    return x1 - x2


def returnStd(std1, std2, n1, n2):
    return math.sqrt((std1 ** 2) / n1 + (std2 ** 2) / n2)


def returnVar(std1, std2, n1, n2):
    return math.pow(returnStd(std1, std2, n1, n2), 2)
