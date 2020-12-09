import pandas as pd
import numpy as np
import math
import scipy.stats as stats


def p_value_method(x_mean, h0_mean, h0_std, samp_num, siglevel, option='left', precision=4):
    z_value = (x_mean - h0_mean) / (h0_std/(samp_num ** 0.5))
    opt = option.lower()[0]
    if opt == 'l':
        option = 'One-Tail Test (left tail)'
        p_value = stats.norm.cdf(z_value)
        zcv = stats.norm.ppf(1 - siglevel)
    elif opt == 'r':
        option = 'One-Tail Test (right tail)'
        p_value = stats.norm.sf(z_value)
        zcv = stats.norm.ppf(1 - siglevel)
    elif opt == 't':
        # two-tail test
        option = 'Two-Tail Test'
        p_value = (1 - stats.norm.cdf(z_value)) * 2
        zcv = stats.norm.ppf(1 - siglevel/2)

    if p_value >= 0 and p_value < 0.01:
        inter_p = 'Overwhelming Evidence'
    elif p_value >= 0.01 and p_value < 0.05:
        inter_p = 'Strong Evidence'
    elif p_value >= 0.05 and p_value < 0.1:
        inter_p = 'Weak Evidence'
    elif p_value >= .1:
        inter_p = 'No Evidence'

    result = f"""======= Analysis =======
Mean = {x_mean:.{precision}f}
Number of Observation = {samp_num:.{precision}f}
Hypothesized Mean (H0 Mean) = {h0_mean:.{precision}f}
Assumed Standard Devation = {h0_std:.{precision}f}
Significant Level (alpha) = {siglevel:.{precision}f}

Using {option}:
Difference = {x_mean - h0_mean}
z (Critical value) = {zcv:.{precision}f}
z (Observed value) = {z_value:.{precision}f}
p-value = {p_value:.{precision}f} ({inter_p})"""

    print(result)

    return zcv, p_value
