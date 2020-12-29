from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import scipy.stats as stats


def con_level(alpha, n, s, sigma, show=True, ignore=True):
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
