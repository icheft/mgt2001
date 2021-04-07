from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import scipy.stats as stats


def inter_p_value(p_value):
    # interpretation
    if p_value >= 0 and p_value < 0.01:
        inter_p = 'Overwhelming Evidence'
    elif p_value >= 0.01 and p_value < 0.05:
        inter_p = 'Strong Evidence'
    elif p_value >= 0.05 and p_value < 0.1:
        inter_p = 'Weak Evidence'
    elif p_value >= .1:
        inter_p = 'No Evidence'
    return inter_p


def ranksum_z_test(df=None, to_compute='', alternative=None, precision=4, alpha=0.05):
    """
    df can only have two columns and df.shape[0] > 10
    alternative has three options: 'two-sided', 'less', 'greater'
    """
    # sort all data points by values
    tmp_values = df.values.reshape(-1)
    tmp_values = tmp_values[~np.isnan(tmp_values)]
    tmp_values.sort()

    # assign ranks
    updated_df = pd.DataFrame({'value': tmp_values})
    updated_df['rank'] = updated_df.index + 1

    # average rank for identical value
    updated_df = updated_df.groupby('value').mean().reset_index()
    # display(updated_df)

    # Compute Sum of Ranks
    samp1 = pd.DataFrame({'value': df[to_compute].dropna().values})
    samp1 = pd.merge(samp1, updated_df)
    T = samp1['rank'].sum()

    # compute mean and standard deviation
    n1 = df.iloc[:, 0].dropna().shape[0]
    n2 = df.iloc[:, 1].dropna().shape[0]

    E_T = n1*(n1+n2+1)/2

    sigmaT = (n1*n2*(n1+n2+1)/12) ** 0.5
    z = (T-E_T)/sigmaT
    # compute p-value
    # right (greater)
    p_value = 1 - stats.norm.cdf(z)

    if alternative == 'greater':
        pass
    elif alternative == 'less':
        p_value = stats.norm.cdf(z)
    elif alternative == 'two-sided':
        # two-tail
        if p_value > 0.5:
            p_value = stats.norm.cdf(z)
        p_value *= 2
    flag = False
    if p_value < alpha:
        flag = True

    result = f'''======= z-test =======
T (sum of ranks) = {T}
(n1, n2) = ({n1}, {n2})
mu_t = {E_T}
sigma_t = {sigmaT}
z statistic value (observed) = {z:.{precision}f}
p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})
Reject H_0 ({alternative}) → {flag}
'''
    print(result)
    result_dict = {'T': T, 'ET': E_T,
                   'sigmaT': sigmaT, 'z': z, 'p-value': p_value}
    return updated_df, result_dict


def sign_binom_test(diff=None, sign='+', alternative=None, precision=4, alpha=0.05):
    n = diff.size - np.sum(diff == 0)

    if sign == '+':
        sign_count = np.sum(diff > 0)
    else:
        sign_count = np.sum(diff < 0)

    if alternative == 'greater' or alternative == 'less':
        # 如果超過一半就要切換
        if sign_count > n / 2:
            p_value = 1 - stats.binom.cdf(sign_count - 1, n=n, p=0.5)
        else:
            p_value = stats.binom.cdf(sign_count, n=n, p=0.5)
    elif alternative == 'two-sided':
        p_value = stats.binom.cdf(sign_count, n=n, p=0.5)
        if p_value > 0.5:
            p_value = 1 - stats.binom.cdf(sign_count - 1, n=n, p=0.5)

        p_value *= 2

    flag = False
    if p_value < alpha:
        flag = True

    result = f'''======= Sign Test - Binomial Distribution =======
(For small sample size (<= 10))

Targeted Sign: {sign}
n = {n}
Sign counts = {sign_count}

p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})
Reject H_0 ({alternative}) → {flag}
    '''
    print(result)
    return sign_count, p_value


def sign_z_test(diff=None, sign='+', alternative=None, precision=4, alpha=0.05):
    diff = diff[~(diff == 0)]
    n = len(diff)

    if sign == '+':
        T = np.sum(diff > 0)
    else:
        T = np.sum(diff < 0)
    z_stat = (T - 0.5 * n) / (.5 * (n ** 0.5))
    # right tail
    if alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z_stat)
    elif alternative == 'less':
        p_value = stats.norm.cdf(z_stat)
    elif alternative == 'two-sided':
        p_value = 1 - stats.norm.cdf(z_stat)
        if p_value > 0.5:
            p_value = stats.norm.cdf(z_stat)
        p_value *= 2
    flag = False
    if p_value < alpha:
        flag = True
    result = f'''======= Sign Test - z Statistic =======
(For large sample size (> 10))

Targeted Sign: {sign}
n = {n}
Sign counts = {T}

z statistic = {z_stat:.{precision}f}
p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})
Reject H_0 ({alternative}) → {flag}
    '''
    print(result)

    return T, p_value


def wilcoxon_signed_ranksum_z_test(diff=None, sign='+', alternative=None, precision=4, alpha=0.05):

    diff = diff[~(diff == 0)]
    n = len(diff)

    diff_abs = np.sort(np.abs(diff).to_numpy())

    updated_diff = pd.DataFrame({'diff_abs': diff_abs})
    updated_diff['rank'] = updated_diff.index + 1
    updated_diff = updated_diff.groupby('diff_abs').mean().reset_index()

    new_df = pd.DataFrame({'diff': diff, 'diff_abs': np.abs(diff)})
    new_df = pd.merge(new_df, updated_diff)

    if sign == '+':
        T = np.sum(new_df['rank'][new_df['diff'] > 0])
    else:
        T = np.sum(new_df['rank'][new_df['diff'] < 0])

    E_T = n * (n + 1) / 4
    sigma_T = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5

    z_stat = (T - E_T) / sigma_T

    if alternative == 'greater':
        # right tail test
        p_value = 1 - stats.norm.cdf(z_stat)
    elif alternative == 'less':
        # left tail test
        p_value = stats.norm.cdf(z_stat)
    elif alternative == 'two-sided':
        # two-tailed test
        p_value = 1 - stats.norm.cdf(z_stat)
        if p_value > 0.5:
            p_value = stats.norm.cdf(z_stat)
        p_value *= 2

    flag = False
    if p_value < alpha:
        flag = True

    result = f'''======= Wilcoxon Signed Rank Sum Test - z Statistic =======
(For large sample size (> 30))

Targeted Sign: {sign}
n = {n}
Sum of rank (T statistic) = {T}

mu_t = {E_T}
sigma_t = {sigma_T}

z statistic value (observed) = {z_stat:.{precision}f}
p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})
Reject H_0 ({alternative}) → {flag}
    '''
    print(result)

    result_dict = {'n': n, 'T': T, 'E_T': E_T,
                   'sigma_T': sigma_T, 'z_stat': z_stat, 'p_value': p_value}

    return new_df, result_dict
