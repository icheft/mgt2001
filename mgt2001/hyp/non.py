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


def grank(data):
    if type(data) == np.ndarray or type(data) == list:
        alldata = data.copy()
        data = data.copy()
    else:
        alldata = data.values.copy()
        data = data.values.copy()
    alldata.sort()
    tmp_df = pd.DataFrame({'value': alldata})
    tmp_df['rank'] = tmp_df.index + 1
    value_to_rank = tmp_df.groupby('value').mean().reset_index()
    samp = pd.DataFrame({'value': data})
    samp = pd.merge(samp, value_to_rank, how='left')
    return samp['rank']


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


def kruskal_chi2_test(data=None, alpha=0.05, precision=4):
    """
    col = 要比較的 target
    row = data for each target
    """
    if type(data) == pd.DataFrame:
        data = data.copy().to_numpy()
        alldata = np.concatenate(data.copy())
    else:
        alldata = np.concatenate(data.copy())

    k = data.shape[1]
    alldata.sort()

    tmp_df = pd.DataFrame(({'value': alldata}))
    tmp_df['rank'] = tmp_df.index + 1  # rank
    value_to_rank = tmp_df.groupby('value').mean().reset_index()
    T = []
    sample_rank_df = []
    for i in range(k):

        samp = pd.DataFrame(
            {'value': data[:, i][~np.isnan(data[:, i])]})

        samp = pd.merge(samp, value_to_rank)
        sample_rank_df.append(samp)
        T.append(samp['rank'].sum())

    n = [len(data[:, i][~np.isnan(data[:, i])]) for i in range(k)]

    # print(T)
    # print(n)

    rule_of_five_str = ""
    if (np.sum(np.array(n) < 5) > 0):
        rule_of_five_str += "!(At least one sample size is less than 5)"
    else:
        rule_of_five_str += "(All sample size >= 5)"

    N = np.sum(n)

    t_over_n = 0

    for i in range(k):
        t_over_n += T[i] ** 2 / n[i]

    H = 12 / N / (N + 1) * t_over_n - 3 * (N + 1)
    p_value = 1 - stats.chi2.cdf(H, k - 1)
    chi2_stat = stats.chi2.ppf(1 - alpha, k - 1)

    result_dict = {'H': H, 'p-value': p_value,
                   'T': T, 'sample_rank_df': sample_rank_df}
    flag = p_value < alpha

    result = f'''======= Kruskal-Wallis Test with Chi-squared Test =======
{rule_of_five_str}

H statistic value (observed) = {H:.{precision}f}
chi2 critical value = {chi2_stat:.{precision}f}
p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})
Reject H_0 (Not all {k} population locations are the same) → {flag}
    '''
    print(result)
    return result_dict


def friedman_chi2_test(data=None, alpha=0.05, precision=4):
    """
    col = 要比較的 target
    row = blocked data for each target
    """
    if type(data) == np.ndarray:
        data = pd.DataFrame(data)

    new_df = data.apply(grank, axis=1)
    b, k = new_df.shape

    rule_of_five_str = ""
    if (b < 5 and k < 5):
        rule_of_five_str += f"!(Number of blocks = {b} < 5 and number of populations = {k} < 5)"
    else:
        rule_of_five_str += f"(Number of blocks = {b} >= 5 or number of populations {k} >= 5)"

    T = new_df.sum().to_numpy()

    F_r = 12 / b / k / (k + 1) * np.sum(T ** 2) - 3 * b * (k + 1)
    p_value = 1 - stats.chi2.cdf(F_r, k - 1)
    chi2_stat = stats.chi2.ppf(1 - alpha, k - 1)

    result_dict = {'F_r': F_r, 'p-value': p_value,
                   'T': T, 'sample_ranked_df': new_df}
    flag = p_value < alpha

    result = f'''======= Friedman Test with Chi-squared Test =======
{rule_of_five_str}

F_r statistic value (observed) = {F_r:.{precision}f}
chi2 critical value = {chi2_stat:.{precision}f}
p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})
Reject H_0 (Not all {k} population locations are the same) → {flag}
    '''
    print(result)
    return result_dict


def pearson_test(data=None, a=None, b=None, alpha=0.05, precision=4):
    """
    a, b 還不能傳入東西
    Make sure that data is in the form of [a, b]
    """
    cov_mat = np.cov(data.values, rowvar=False)
    cor_mat = np.corrcoef(data.values, rowvar=False)
    cov = cov_mat[0][1]
    cor = cor_mat[0][1]

    n = data.shape[0]
    d_of_f = n - 2
    t_c = stats.t.ppf(1 - alpha / 2, df=d_of_f)
    t_stat = cor * (((n - 2) / (1 - cor ** 2)) ** 0.5)

    flag = abs(t_stat) > t_c
    result_dict = {'cov': cov, 't_stat': t_stat, 'cor': cor, 't_c': t_c}
    results = f"""======= Pearson Correlation Coefficient =======
Covariance: {cov:.{precision}f}
Coefficient of Correlation: {cor:.{precision}f}

t (Critical Value) = {t_c:.{precision}f}
t (Observed Value) = {t_stat:.{precision}f}

Reject H_0 (There are linear relationship between two variables) → {flag}
"""

    print(results)

    return result_dict


def spearman_test(a=None, b=None, alpha=0.05, precision=4):
    spearman_restult_cor, spearman_restult_p_value = stats.spearmanr(a, b)
    # print(f'Correlation = {cor:.4f}, p-value={p_value:.4f}')
    n = len(a)

    rule_of_30_str = ''

    results = f"""======= Spearman Rank Correlation Coefficient =======
[scipy.stats.spearmanr]
Coefficient of Correlation: {spearman_restult_cor:.{precision}f}
p-value={spearman_restult_p_value:.{precision}f} ({inter_p_value(spearman_restult_p_value)})
"""

    if (n < 30):
        rule_of_30_str += f"!(n = {n} < 30)"
        flag = spearman_restult_p_value < alpha
        results += f"""
Reject H_0 (There are relationship between two variables) → {flag}
        """
        result_dict = {'spearman_result': [
            spearman_restult_cor, spearman_restult_p_value]}
    else:
        rule_of_30_str += f"(n = {n} >= 30)"
        flag = spearman_restult_p_value < alpha
        results += f"""
Reject H_0 (There are relationship between two variables) → {flag}
        """
        z_stat = spearman_restult_cor * ((n - 1) ** 0.5)
        z_cv = stats.norm.ppf(1 - alpha/2)
        p_value = stats.norm.sf(z_stat) * 2
        if p_value > 1:
            p_value = stats.norm.cdf(z_stat) * 2
        flag = p_value < alpha
        results += f"""
[z test statistic]
{rule_of_30_str}

r_s: {spearman_restult_cor:.{precision}f} (using spearmanr's result)
z stat (observed value) = {z_stat:.{precision}f}
z (critical value) = {z_cv:.{precision}f}
p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})
Reject H_0 (There are relationship between two variables) → {flag}
        """

        result_dict = {'spearman_result': [
            spearman_restult_cor, spearman_restult_p_value], 'z_stat': z_stat, 'z_cv': z_cv, 'p-value': p_value}

    print(results)

    return result_dict
