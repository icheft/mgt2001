from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import scipy.stats as stats

from .. import samp


def samp_size(s_p, width, alpha):
    """
    Input: s_p (sample proportion), width, alpha
    Output: Estimated sample size
    """
    z_cv = stats.norm.ppf(1 - alpha / 2)
    return (z_cv * math.sqrt(s_p * (1 - s_p)) / width) ** 2


def con_level(s_p, n, alpha, show=True, Wilson=False, N=False, correction=True):
    """
    Caution: np̂ > 5 and n(1 - p̂) > 5
    Input: s_p, n, alpha, show=True, Wilson=False, N=False, correction=True
    Output: {"lat": lat, "lcl": lcl, "ucl": ucl, "z_cv": z_cv}

    Note:
        z_cv = stats.norm.ppf(1 - alpha / 2)
        lat = z_cv * math.sqrt(s_p * (1 - s_p)/n)
        or for Wilson: lat = z_cv * math.sqrt(s_p * (1 - s_p)/(n + 4))

    Just in case:
    if there is no need for correction, but is is corrected
    go through 'lat' and do:
        lcl = s_p - lat
        ucl = s_p + lat
    """
    con_coef = 1 - alpha
    z_cv = stats.norm.ppf(1 - alpha / 2)
    if not Wilson and not samp.check5(n, s_p):
        print('Not satisfying np̂ > 5 and n(1 - p̂) > 5...')
    if Wilson:
        # make sure that you have arrange s_p to (x + 2) / (n + 4)
        lat = z_cv * math.sqrt(s_p * (1 - s_p)/(n + 4))
    else:
        lat = z_cv * math.sqrt(s_p * (1 - s_p)/n)

    lcl = s_p - lat
    ucl = s_p + lat

    if N:
        if n / N > 0.5 and correction:
            print("Corrected...")
            fpcf = math.sqrt((N - n)/(N - 1))
            lcl = s_p - lat * fpcf
            ucl = s_p + lat * fpcf
        elif correction:
            print("Corrected...")
            fpcf = math.sqrt((N - n)/(N - 1))
            lcl = s_p - lat * fpcf
            ucl = s_p + lat * fpcf
        if lcl < 0:
            lcl = 0
        if ucl < 0:
            ucl = 0
        result = f"""{con_coef * 100:.1f}% Confidence Interval: N [{lcl:.4f}, {ucl:.4f}] = [{N * lcl:.4f}, {N * ucl:.4f}]
p̂: {s_p:.4f}
Sample Size: {n}
z_cv (Critical value): {z_cv:.4f}
    """
    else:
        if lcl < 0:
            lcl = 0
        if ucl < 0:
            ucl = 0
        result = f"""{con_coef * 100:.1f}% Confidence Interval: [{lcl:.4f}, {ucl:.4f}]
p̂: {s_p:.4f}
Sample Size: {n}
z_cv (Critical value): {z_cv:.4f}
    """
    if show:
        print(result)
    return {"lat": lat, "lcl": lcl, "ucl": ucl, "z_cv": z_cv}


def rejection_region_method(s_p, h0_p, nsize, alpha, option='left', precision=4, show=True, ignore=False):
    """
    Input: s_p, h0_p, nsize, alpha, option='left', precision=4, show=True, ignore=False
    Output: 
        if opt == 't':
            return p_l, p_u
        else:
            return p_c
    """
    opt = option.lower()[0]
    if not samp.check5(nsize, h0_p):
        print('Not satisfying np_0 > 5 and n(1 - p_0) > 5...')
    if opt == 't':
        option = 'Two-Tail Test'
        z_cv = stats.norm.ppf(1 - alpha/2)
        p_u = h0_p + z_cv * math.sqrt(h0_p * (1 - h0_p)/nsize)
        p_l = h0_p - z_cv * math.sqrt(h0_p * (1 - h0_p)/nsize)
        flag = s_p < p_l or s_p > p_u
        if not ignore:
            result = f'''======= The Rejection Region Method =======
Significant Level (alpha) = {alpha:.{precision}f}
z (Critical value) = {z_cv:.{precision}f}

Using {option}:
p̂ =  {s_p:.{precision}f}
p_l (Lower bound for the critical value) = {p_l:.{precision}f}
p_u (Upper bound for the critical value) = {p_u:.{precision}f}
Reject H_0 → {flag}
            '''
        else:
            result = f'''======= The Rejection Region Method =======
Significant Level (alpha) = {alpha:.{precision}f}
z (Critical value) = {z_cv:.{precision}f}

Using {option}:
p_l (Lower bound for the critical value) = {p_l:.{precision}f}
p_u (Upper bound for the critical value) = {p_u:.{precision}f}
            '''

    else:
        z_cv = stats.norm.ppf(1 - alpha)
        if opt == 'l':
            # left tail
            option = 'One-Tail Test (left tail)'
            p_c = h0_p - z_cv * math.sqrt(h0_p * (1 - h0_p)/nsize)
            flag = s_p < p_c
        elif opt == 'r':
            option = 'One-Tail Test (right tail)'
            p_c = h0_p + z_cv * math.sqrt(h0_p * (1 - h0_p)/nsize)
            flag = s_p > p_c
        if not ignore:
            result = f'''======= The Rejection Region Method =======
Significant Level (alpha) = {alpha:.{precision}f}
z (Critical value) = {z_cv:.{precision}f}

Using {option}:
p̂ =  {s_p:.{precision}f}
p_c (Critical value) = {p_c:.{precision}f}
Reject H_0 → {flag}
            '''
        else:
            result = f'''======= The Rejection Region Method =======
Significant Level (alpha) = {alpha:.{precision}f}
z (Critical value) = {z_cv:.{precision}f}

Using {option}:
p_c (Critical value) = {p_c:.{precision}f}
            '''

    if show:
        print(result)

    if opt == 't':
        return p_l, p_u
    else:
        return p_c


def testing_statistic_method(s_p, h0_p, nsize, alpha, option='left', precision=4, ignore=False):
    """
    Input: s_p, h0_p, nsize, alpha, option='left', precision=4, ignore=False
    Output: z_stats, z_cv
    """
    opt = option.lower()[0]
    z_stats = (s_p - h0_p)/math.sqrt(h0_p * (1 - h0_p)/nsize)
    if not samp.check5(nsize, h0_p):
        print('Not satisfying np_0 > 5 and n(1 - p_0) > 5...')

    if opt == 't':
        z_cv = stats.norm.ppf(1 - alpha / 2)
        option = 'Two-Tail Test'
        flag = z_stats < -z_cv or z_stats > z_cv

        if not ignore:
            result = f'''======= Testing Statistic Method =======
Significant Level (alpha) = {alpha:.{precision}f}
z_cv (Critical value) = {z_cv:.{precision}f}

Using {option}:
z_stats (Observed value) =  {z_stats:.{precision}f}
-z_cv (Lower bound for the critical value) = {-z_cv:.{precision}f}
z_cv (Upper bound for the critical value) = {z_cv:.{precision}f}
Reject H_0 → {flag}
            '''
        else:
            result = f'''======= Testing Statistic Method =======
Significant Level (alpha) = {alpha:.{precision}f}
z_cv (Critical value) = {z_cv:.{precision}f}

Using {option}:
-z_cv (Lower bound for the critical value) = {-z_cv:.{precision}f}
z_cv (Upper bound for the critical value) = {z_cv:.{precision}f}
            '''

    else:
        z_cv = stats.norm.ppf(1 - alpha)
        if opt == 'l':
            # left tail
            option = 'One-Tail Test (left tail)'
            z_cv = -z_cv
            flag = z_stats < z_cv
        elif opt == 'r':
            option = 'One-Tail Test (right tail)'
            flag = z_stats > z_cv

        if not ignore:
            result = f'''======= Testing Statistic Method =======
Significant Level (alpha) = {alpha:.{precision}f}
z_cv (Critical value) = {z_cv:.{precision}f}

Using {option}:
z_stats (Observed value) =  {z_stats:.{precision}f}
z_cv (Critical value) = {z_cv:.{precision}f}
Reject H_0 → {flag}
            '''

        else:
            result = f'''======= Testing Statistic Method =======
Significant Level (alpha) = {alpha:.{precision}f}
z_cv (Critical value) = {z_cv:.{precision}f}

Using {option}:
z_cv (Critical value) = {z_cv:.{precision}f}
            '''

    print(result)

    return z_stats, z_cv


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


def p_value_method(s_p, h0_p, nsize, alpha, option='left', precision=4):
    """
    Input: s_p, h0_p, nsize, alpha, option='left', precision=4
    Output: z_cv, z_stats, p_value
    """
    opt = option.lower()[0]
    z_stats = (s_p - h0_p)/math.sqrt(h0_p * (1 - h0_p)/nsize)
    if not samp.check5(nsize, h0_p):
        print('Not satisfying np_0 > 5 and n(1 - p_0) > 5...')
    if opt == 't':
        # two-tail test
        option = 'Two-Tail Test'
        if s_p > h0_p:
            p_value = stats.norm.sf(z_stats) * 2
        else:
            p_value = stats.norm.cdf(z_stats) * 2

        z_cv = stats.norm.ppf(1 - alpha/2)
        flag = p_value < alpha
        sub_result = f'''Using {option}:
Difference = {s_p - h0_p}
z_cv (Critical value) = {-z_cv:.{precision}f}, {z_cv:.{precision}f}
z_stats (Observed value) = {z_stats:.{precision}f}
p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})
Reject H_0 → {flag}
        '''
    else:
        if opt == 'l':
            option = 'One-Tail Test (left tail)'
            p_value = stats.norm.cdf(z_stats)
            z_cv = -stats.norm.ppf(1 - alpha)
        elif opt == 'r':
            option = 'One-Tail Test (right tail)'
            p_value = stats.norm.sf(z_stats)
            z_cv = stats.norm.ppf(1 - alpha)
        flag = p_value < alpha
        sub_result = f'''Using {option}:
Difference = {s_p - h0_p}
z_cv (Critical value) = {z_cv:.{precision}f}
z_stats (Observed value) = {z_stats:.{precision}f}
p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})
Reject H_0 → {flag}
        '''

    result = f"""======= p-value Method =======
p̂ = {s_p:.{precision}f}
Number of Observation = {nsize:.{precision}f}
Hypothesized Proportion (H0 Mean) = {h0_p:.{precision}f}
Significant Level (alpha) = {alpha:.{precision}f}

""" + sub_result

    print(result)

    return z_cv, z_stats, p_value
