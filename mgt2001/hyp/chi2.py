from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import scipy.stats as stats


def con_level(alpha, n, s, sigma, show=True, ignore=True):
    """
Input: alpha, n (sample size), s (sample standard deviation), sigma (population standard deviation, can ignore if ignore=True - default), show=True
Return the confidence level at alpha. Return a dictionary: {"lcl": lcl, "ucl": ucl}

+ `show`: default is `True`. Set to `False` to disable rendering.
    """
    df_v = n - 1
    con_coef = 1 - alpha

    chi2_cv_u = stats.chi2.ppf(1 - alpha / 2, df=df_v)
    chi2_cv_l = stats.chi2.ppf(alpha / 2, df=df_v)

    lcl = (df_v) * (s ** 2) / chi2_cv_u
    ucl = (df_v) * (s ** 2) / chi2_cv_l

    result = f"""{con_coef * 100:.1f}% Confidence Interval: [{lcl:.4f}, {ucl:.4f}]

S^2 (Sample Variance): {s ** 2:.4f}
Sample Size: {n}
"""
    if ignore == False:
        flag = sigma ** 2 >= lcl and sigma ** 2 <= ucl
        if flag:
            cover = "covers the real/population variance."
        else:
            cover = "does not cover the real/population variance."

        result += f'''sigma^2 (Population Variance) = {sigma ** 2:.4f}
The interval {cover}
'''
    if show:
        print(result)
    return {"lcl": lcl, "ucl": ucl}


def rejection_region_method(alpha, n, s, sigma, option='left', precision=4, show=True, ignore=False):
    """
    Input: alpha, n, s, sigma, option='left', precision=4, show=True, ignore=False
    Output: 
        if opt == 't':
            return s2_l, s2_u
        else:
            return s2_c
    """
    opt = option.lower()[0]
    df_v = n - 1
    var = s ** 2
    if opt == 't':
        option = 'Two-Tail Test'
        chi2_cv_u = stats.chi2.ppf(1 - alpha / 2, df=df_v)
        chi2_cv_l = stats.chi2.ppf(alpha / 2, df=df_v)
        s2_u = chi2_cv_u * (sigma ** 2) / df_v
        s2_l = chi2_cv_l * (sigma ** 2) / df_v
        flag = var < s2_l or var > s2_u
        if not ignore:
            result = f'''======= The Rejection Region Method =======
Significant Level (alpha) = {alpha:.{precision}f}
chi2_l (Lower bound for Chi2 critical value) = {chi2_cv_l:.{precision}f}
chi2_u (Upper bound for Chi2 critical value) = {chi2_cv_u:.{precision}f}

Using {option}:
S^2 (var) =  {var:.{precision}f}
S_L^2 (Lower bound for the critical value) = {s2_l:.{precision}f}
S_U^2 (Upper bound for the critical value) = {s2_u:.{precision}f}
Reject H_0 → {flag}
            '''
        else:
            result = f'''======= The Rejection Region Method =======
Significant Level (alpha) = {alpha:.{precision}f}
chi2_l (Lower bound for Chi2 critical value) = {chi2_cv_l:.{precision}f}
chi2_u (Upper bound for Chi2 critical value) = {chi2_cv_u:.{precision}f}

Using {option}:
S_L^2 (Lower bound for the critical value) = {s2_l:.{precision}f}
S_U^2 (Upper bound for the critical value) = {s2_u:.{precision}f}
            '''

    else:
        if opt == 'l':
            # left tail
            option = 'One-Tail Test (left tail)'
            chi2_value = df_v * (s ** 2) / (sigma ** 2)
            chi2_cv = stats.chi2.ppf(alpha, df_v)
            s2_c = chi2_cv * (sigma ** 2) / df_v
            flag = var < s2_c
        elif opt == 'r':
            option = 'One-Tail Test (right tail)'
            chi2_value = df_v * (s ** 2) / (sigma ** 2)
            chi2_cv = stats.chi2.ppf(1 - alpha, df_v)
            s2_c = chi2_cv * (sigma ** 2) / df_v
            flag = var > s2_c
        if not ignore:
            result = f'''======= The Rejection Region Method =======
Significant Level (alpha) = {alpha:.{precision}f}
chi2 (Critical value) = {chi2_cv:.{precision}f}

Using {option}:
S^2 (var) =  {var:.{precision}f}
S_C^2 (Critical value) = {s2_c:.{precision}f}
Reject H_0 → {flag}
            '''
        else:
            result = f'''======= The Rejection Region Method =======
Significant Level (alpha) = {alpha:.{precision}f}
chi2 (Critical value) = {chi2_cv:.{precision}f}

Using {option}:
S_C^2 (Critical value) = {s2_c:.{precision}f}
            '''

    if show:
        print(result)

    if opt == 't':
        return s2_l, s2_u
    else:
        return s2_c


def testing_statistic_method(alpha, n, s, sigma, option='left', precision=4, ignore=False):
    """
    Input: alpha, n, s, sigma, option='left', precision=4, show=True, ignore=False
    Output: 
        if opt == 't':
            return chi2_value, chi2_cv_l, chi2_cv_u
        else:
            return chi2_value, chi2_cv
    """
    opt = option.lower()[0]
    df_v = n - 1
    var = s ** 2
    chi2_value = df_v * (s ** 2) / (sigma ** 2)

    if opt == 't':
        option = 'Two-Tail Test'
        chi2_cv_u = stats.chi2.ppf(1 - alpha / 2, df=df_v)
        chi2_cv_l = stats.chi2.ppf(alpha / 2, df=df_v)
        flag = chi2_value < chi2_cv_l or chi2_value > chi2_cv_u

        if not ignore:
            result = f'''======= Testing Statistic Method =======
Significant Level (alpha) = {alpha:.{precision}f}

Using {option}:
Chi2 (Observed value) =  {chi2_value:.{precision}f}
Chi2_L (Lower bound for Chi2 critical value) = {chi2_cv_l:.{precision}f}
Chi2_U (Upper bound for Chi2 critical value) = {chi2_cv_u:.{precision}f}
Reject H_0 → {flag}
            '''
        else:
            result = f'''======= Testing Statistic Method =======
Significant Level (alpha) = {alpha:.{precision}f}

Using {option}:
Chi2_L (Lower bound for Chi2 critical value) = {chi2_cv_l:.{precision}f}
Chi2_U (Upper bound for Chi2 critical value) = {chi2_cv_u:.{precision}f}
            '''

    else:
        if opt == 'l':
            # left tail
            option = 'One-Tail Test (left tail)'
            chi2_cv = stats.chi2.ppf(alpha, df_v)
            flag = chi2_value < chi2_cv
        elif opt == 'r':
            option = 'One-Tail Test (right tail)'
            chi2_cv = stats.chi2.ppf(1 - alpha, df_v)
            flag = chi2_value > chi2_cv

        if not ignore:
            result = f'''======= Testing Statistic Method =======
Significant Level (alpha) = {alpha:.{precision}f}

Using {option}:
Chi2 (Observed value) =  {chi2_value:.{precision}f}
Chi2_C (Critical value) = {chi2_cv:.{precision}f}
Reject H_0 → {flag}
            '''

        else:
            result = f'''======= Testing Statistic Method =======
Significant Level (alpha) = {alpha:.{precision}f}

Using {option}:
Chi2_C (Critical value) = {chi2_cv:.{precision}f}
            '''

    print(result)
    if opt == 't':
        return chi2_value, chi2_cv_l, chi2_cv_u
    else:
        return chi2_value, chi2_cv


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


def p_value_method(alpha, n, s, sigma, option='left', precision=4):
    """
    Input: alpha, n, s, sigma, option='left', precision=4
    Output: chi2_stat, p_value
    """
    opt = option.lower()[0]
    df_v = n - 1
    var = s ** 2
    chi2_stat = df_v * (s ** 2) / (sigma ** 2)

    if opt == 't':
        # two-tail test
        option = 'Two-Tail Test'
        alphas = np.arange(0, 1, 0.001)
        chi2s = stats.chi2.ppf(alphas, df_v)
        med = np.median(chi2s)

        if chi2_stat > med:
            p_value = 2 * stats.chi2.sf(chi2_stat, df_v)
        else:
            p_value = 2 * stats.chi2.cdf(chi2_stat, df_v)

        flag = p_value < alpha
        sub_result = f'''Using {option}:
Chi2 Median under d.f. {df_v}: {med:.{precision}f}
Chi2 Stat (Observed value) = {chi2_stat:.{precision}f}
p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})
Reject H_0 → {flag}
        '''
    else:
        if opt == 'l':
            option = 'One-Tail Test (left tail)'
            p_value = stats.chi2.cdf(chi2_stat, df_v)
            chi2_cv = stats.chi2.ppf(alpha, df_v)
        elif opt == 'r':
            option = 'One-Tail Test (right tail)'
            p_value = stats.chi2.sf(chi2_stat, df_v)
            chi2_cv = stats.chi2.ppf(1 - alpha, df_v)
        flag = p_value < alpha
        sub_result = f'''Using {option}:
Chi2_C (Critical value) = {chi2_cv:.{precision}f}
Chi2 Stat (Observed value) = {chi2_stat:.{precision}f}
p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})
Reject H_0 → {flag}
        '''

    result = f"""======= p-value Method =======
S^2 (var) = {var:.{precision}f}
Number of Observation = {n:.{precision}f}
Hypothesized Std. (H0 Std.) = {sigma:.{precision}f}
Sample Standard Deviation = {s:.{precision}f}
Significant Level (alpha) = {alpha:.{precision}f}

""" + sub_result

    print(result)

    return chi2_stat, p_value

# Chi-Squared Tests for two or more variables


def multinomial(freq_o, freq_e, alpha=0.05, precision=4):
    """
    >>> return chi2_stat, chi2_cv, p_value
    Check RULE of FIVE before running the test
    >>> if np.sum(freq_e < 5) > 0:
    >>>    print("Rule of five is not met. ")
    """
    # if np.sum(freq_e < 5) > 0:
    #     print("Rule of five is not met.")
    #     return
    stat, p_value = stats.chisquare(freq_o, freq_e)
    df = freq_o.shape[0]-1
    chi2_cv = stats.chi2.ppf(1 - alpha, df)
    flag = False
    if p_value < 0.05:
        flag = True
    result = f"""======= Goodness of Fit Test: A Multinomial Population =======
Chi-squared test: 
chi2 statistics = {stat:.{precision}f}
chi2 critical value = {chi2_cv:.{precision}f}
Degree of freedom = {df}
p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})

Reject H_0 (does not follow the probability) → {flag}
"""
    print(result)
    return stat, chi2_cv, p_value


def contingency(cont_table=None, alpha=0.05, precision=4):
    """
    Check RULE of FIVE before running the test
    >>> return chi2, p, dof, ex
    """
    chi2, p, dof, ex = stats.chi2_contingency(cont_table, correction=False)
    chi2_cv = stats.chi2.ppf(1 - alpha, dof)
    flag = False
    if p < alpha:
        flag = True

    result = f"""======= Tests of Independence: Contingency Table =======
chi2 Statistics = {chi2:.{precision}f}
chi2 critical value = {chi2_cv:.{precision}f}

Degree of freedom = {dof}
p-value = {p:.{precision}f} ({inter_p_value(p)})

Reject H_0 (dependent) → {flag}

Expected Frequency:
{ex}
"""
    print(result)
    return chi2, p, dof, ex

# Poisson


def frequency_table(interval, observed, nobs, total_time, mu):
    o_vs_e_freq_df = pd.DataFrame(columns=['i', 'f_i', 'e_i', 'f_i - e_i'])
    poisson_table = stats.poisson(mu)
    N = total_time
    for (i, f) in zip(interval, observed):
        e = poisson_table.pmf(i)
        e *= N
        o_vs_e_freq_df = o_vs_e_freq_df.append(
            {'i': i, 'f_i': f, 'e_i': e, 'f_i - e_i': f - e}, ignore_index=True)
    return o_vs_e_freq_df


def df_combine(df=None, to_combine=None):
    """
    >>> to_combine = [[0,1,2],[10, 11, 12]]
    >>> updated_df = df_combine(df=o_vs_e_freq_df, to_combine=to_combine)
    """
    if len(to_combine) == 0:
        return df
    else:
        updated_df = pd.DataFrame(columns=['i', 'f_i', 'e_i', 'f_i - e_i'])
        n = len(to_combine)
        for i in range(n):
            if i == 0:
                current_end = to_combine[i][-1]
                if i + 1 < n:
                    next_start = to_combine[i + 1][0]
                else:
                    next_start = None
                updated_df = updated_df.append(
                    df.iloc[to_combine[i]].sum(axis=0), ignore_index=True)
                updated_df.i[i] = f'{to_combine[i]}'
                updated_df = updated_df.append(
                    df.iloc[current_end + 1:next_start], ignore_index=True)
            else:
                current_start = to_combine[i][0]
                prev_end = to_combine[i - 1][-1]
                current_end = to_combine[i][-1]
                if i + 1 < n:
                    next_start = to_combine[i + 1][0]
                else:
                    next_start = None
                updated_df = updated_df.append(
                    df.iloc[to_combine[i]].sum(axis=0), ignore_index=True)
                updated_df.iloc[current_start - prev_end +
                                (i - 1), 0] = f'{to_combine[i]}'
                updated_df = updated_df.append(
                    df.iloc[current_end + 1:next_start], ignore_index=True)

        return updated_df


def poisson_test(df, alpha=0.05, precision=4):
    """
    >>> return chi2_stat, chi2_cv, dof
    """
    dof = df.shape[0] - 1 - 1
    chi2_stat = sum((df['f_i - e_i']) ** 2 / df['e_i'])
    chi2_cv = stats.chi2.ppf(1 - alpha, df=dof)
    flag = False  # cannot reject
    if chi2_stat > chi2_cv:
        flag = True

    result = f"""======= Goodness of Fit Test: Poisson Test =======
chi2 statistic (observed value): {chi2_stat:.{precision}f}
chi2 critical value: {chi2_cv:.{precision}f}
Degree of freedom: {dof}

Reject H_0 (does not follow a Poisson distribution) → {flag}
"""
    print(result)
    return chi2_stat, chi2_cv, dof


def normal_test(srs=None, z_intervals=None, freq_o=None, alpha=0.05, precision=4):
    """
    pass in Series instead of DataFrame
    >>> return o_vs_e_freq_df, chi2_stat, chi2_cv, dof

    Note:
    >>> z_value = stats.norm.ppf(area)
    """
    if z_intervals is not None:
        o_vs_e_freq_df = pd.DataFrame(columns=['i', 'f_i', 'e_i', 'f_i - e_i'])

        # intervals = [(-1), (-1, 0), (0, 1), (1)]
        # freq_o = np.array([6, 27, 14, 3])
        nobs = np.sum(freq_o)

        snd = stats.norm

        for i, interval in enumerate(z_intervals):
            try:
                if len(interval) > 1:
                    z_value = snd.cdf(interval[1]) - snd.cdf(interval[0])
            except:
                if i == 0:
                    z_value = snd.cdf(interval)
                else:
                    z_value = snd.sf(interval)

            e_i = nobs * z_value
            o_vs_e_freq_df = o_vs_e_freq_df.append(
                {'i': interval, 'f_i': freq_o[i], 'e_i': e_i, 'f_i - e_i': freq_o[i] - e_i}, ignore_index=True)

        if np.sum(o_vs_e_freq_df['e_i'] < 5) > 0:
            print("Rule of five is not met. ")

        chi2_stat = np.sum(o_vs_e_freq_df['f_i - e_i']
                           ** 2 / o_vs_e_freq_df['e_i'])
        dof = o_vs_e_freq_df.shape[0] - 2 - 1
        chi2_cv = stats.chi2.ppf(1 - alpha, df=dof)
        flag = False
        if chi2_stat > chi2_cv:
            flag = True
        result = f"""======= Goodness of Fit Test: Normal Distribution =======
chi2 statistics = {chi2_stat:.{precision}f}
chi2 critical value = {chi2_cv:.{precision}f}
Degree of freedom = {dof}

Reject H_0 (reject the assumption that the population is normally distributed) → {flag}
    """
        print(result)

        return o_vs_e_freq_df, chi2_stat, chi2_cv, dof

    elif srs is not None:
        snd = stats.norm
        arr = np.array(srs)
        o_vs_e_freq_df = pd.DataFrame(columns=['i', 'f_i', 'e_i', 'f_i - e_i'])
        n = len(arr)
        mu = arr.mean()
        std = arr.std(ddof=1)
        interval = n / 5
        area = 1 / interval
        acc_area = area
        ctr = 0
        while abs(acc_area - 1) > 0.00000001:
            z_value = snd.ppf(acc_area)
            x_value = z_value * std + mu
            acc_area += area
            f_i = sum(arr <= x_value)
            arr = arr[arr > x_value]

            ctr += 1
            o_vs_e_freq_df = o_vs_e_freq_df.append(
                {'i': x_value, 'f_i': f_i, 'e_i': 5, 'f_i - e_i': f_i - 5}, ignore_index=True)

        num_df = o_vs_e_freq_df.i
        f_i = sum(arr > x_value)
        x_value = f'> {x_value:.3f}'
        o_vs_e_freq_df = o_vs_e_freq_df.append(
            {'i': x_value, 'f_i': f_i, 'e_i': 5, 'f_i - e_i': f_i - 5}, ignore_index=True)

        for i in range(1, ctr):
            o_vs_e_freq_df.iat[i, 0] = f'{num_df[i - 1]:.3f} ~ {num_df[i]:.3f}'
        o_vs_e_freq_df.iat[0, 0] = f'<= {o_vs_e_freq_df.i[0]:.3f}'

        # test
        chi2_stat = np.sum(o_vs_e_freq_df['f_i - e_i']
                           ** 2 / o_vs_e_freq_df['e_i'])
        dof = o_vs_e_freq_df.shape[0] - 2 - 1
        chi2_cv = stats.chi2.ppf(1 - alpha, df=dof)
        flag = False
        if chi2_stat > chi2_cv:
            flag = True
        result = f"""======= Goodness of Fit Test: Normal Distribution =======
chi2 statistics = {chi2_stat:.{precision}f}
chi2 critical value = {chi2_cv:.{precision}f}
Degree of freedom = {dof}

μ = {mu:.{precision}f}
σ = {std:.{precision}f}

Reject H_0 (reject the assumption that the population is normally distributed with μ = {mu:.{precision}f} and σ = {std:.{precision}f}) → {flag}
        """
        print(result)

        return o_vs_e_freq_df, chi2_stat, chi2_cv, dof
