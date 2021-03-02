import pandas as pd
import numpy as np
import math
import scipy.stats as stats
import statsmodels.stats.api as sms


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


def two_population(a, b, alpha=.05, consistency='equal', option='right', show_table=False, stages=[1, 2, 3], show=True):
    """
+ [First stage]: F Statistics - consistency: equal, left (1 is more consistent than 2), right (2 is more consistent than 1)
+ [Second stage]: t Test
+ [Third stage]: Confidence Interval

Will return a result_dict regardless of stages.
    """
    opt = option.lower()[0]
    const = consistency.lower()[0]

    result_dict = dict()

    df_1 = len(a) - 1
    df_2 = len(b) - 1

    varall = [stats.describe(a).variance,
              stats.describe(b).variance]
    f_value = varall[0] / varall[1]

    result_dict['varall'] = varall
    result_dict['f_value'] = f_value

    p_value = 1 - stats.f.cdf(f_value, df_1, df_2)
    if const != 'e':
        p_value /= 2

    if const == 'e':
        rej_upper = stats.f.ppf(1 - alpha/2, df_1, df_2)
        rej_lower = stats.f.ppf(alpha/2, df_1, df_2)
        result_dict['f_rej_upper'] = rej_upper
        result_dict['f_rej_lower'] = rej_lower
        if f_value < rej_lower or f_value > rej_upper:
            flag = True
        else:
            flag = False
        text = 'unequal variances'
    else:
        rej_upper = stats.f.ppf(1 - alpha, df_1, df_2)
        rej_lower = stats.f.ppf(alpha, df_1, df_2)
        if const == 'r':
            result_dict['f_rej_upper'] = rej_upper
            if f_value > rej_upper:
                flag = True
            else:
                flag = False
            text = 'σ_1/σ_2 > 1'
        else:
            result_dict['f_rej_lower'] = rej_lower
            if f_value < rej_lower:
                flag = True
            else:
                flag = False
            text = 'σ_1/σ_2 < 1'

    result_dict['p_value'] = p_value

    results = f"""        1. F Statistics      
==============================
F statistic = {f_value}
p-value = {p_value} ({inter_p_value(p_value)})
Reject H_0 ({text}) → {flag}
"""
    if 2 in stages:
        if flag:  # True == unequal variance
            ttest_result = stats.ttest_ind(a, b, equal_var=False)
            t_summary = list(ttest_result)
            t_critical_two = stats.t.ppf(1 - alpha/2, df=(df_1 + df_2))
            if opt == 'r':
                t_critical_one = stats.t.ppf(1 - alpha, df=(df_1 + df_2))
                result_dict['t_critical_one'] = t_critical_one
            elif opt == 'l':
                t_critical_one = stats.t.ppf(alpha, df=(df_1 + df_2))
                result_dict['t_critical_one'] = t_critical_one

            if opt == 't':
                flag = t_summary[1] < alpha
                result_dict['t_critical_two'] = t_critical_two
                result_dict['t_observed_value'] = t_summary[0]
                result_dict['t_p_value'] = t_summary[1]
                result_dict['df'] = df_1 + df_2
                results += f"""
        2. t Test      
==============================
t (Observed value) = {t_summary[0]}
p-value (two-tail) = {t_summary[1]} ({inter_p_value(t_summary[1])})
t (Critical, two-tail) = {t_critical_two}
DF = {df_1 + df_2}
Reject H_0 → {flag}
"""
            else:
                flag = t_summary[1] / 2 < alpha
                result_dict['t_observed_value'] = t_summary[0]
                result_dict['t_p_value'] = t_summary[1] / 2
                result_dict['df'] = df_1 + df_2
                results += f"""
        2. t Test      
==============================
t (Observed value) = {t_summary[0]}
p-value (one-tail) = {t_summary[1] / 2} ({inter_p_value(t_summary[1] / 2)})
t (Critical, one-tail) = {t_critical_one}
DF = {df_1 + df_2}
Reject H_0 → {flag}
"""
            if 3 in stages:
                cm_result = sms.CompareMeans(
                    sms.DescrStatsW(a), sms.DescrStatsW(b))
                conf_table = cm_result.summary(usevar='unequal', alpha=alpha)
                conf_interval = list(
                    map(float, conf_table.as_text().split('\n')[4].split()[6:]))
                con_coef = 1 - alpha

                # record result
                result_dict['conf_interval'] = conf_interval
                results += f"""
        3. Confidence Interval      
==============================
{con_coef * 100:.1f}% Confidence Interval: [{conf_interval[0]:.4f}, {conf_interval[1]:.4f}]
"""
        else:
            ttest_result = stats.ttest_ind(a, b, equal_var=True)
            t_summary = list(ttest_result)
            t_critical_two = stats.t.ppf(1 - alpha/2, df=(df_1 + df_2))
            if opt == 'r':
                t_critical_one = stats.t.ppf(1 - alpha, df=(df_1 + df_2))
                result_dict['t_critical_one'] = t_critical_one
            elif opt == 'l':
                t_critical_one = stats.t.ppf(alpha, df=(df_1 + df_2))
                result_dict['t_critical_one'] = t_critical_one

            if opt == 't':
                flag = t_summary[1] < alpha
                result_dict['t_critical_two'] = t_critical_two
                result_dict['t_observed_value'] = t_summary[0]
                result_dict['t_p_value'] = t_summary[1]
                result_dict['df'] = df_1 + df_2

                results += f"""
        2. t Test      
==============================
t (Observed value) = {t_summary[0]}
p-value (two-tail) = {t_summary[1]} ({inter_p_value(t_summary[1])})
t (Critical, two-tail) = {t_critical_two}
DF = {df_1 + df_2}
Reject H_0 → {flag}
"""
            else:
                flag = t_summary[1] / 2 < alpha
                result_dict['t_observed_value'] = t_summary[0]
                result_dict['t_p_value'] = t_summary[1]
                result_dict['df'] = df_1 + df_2

                results += f"""
        2. t Test      
==============================
t (Observed value) = {t_summary[0]}
p-value (one-tail) = {t_summary[1] / 2} ({inter_p_value(t_summary[1] / 2)})
t (Critical, one-tail) = {t_critical_one}
DF = {df_1 + df_2}
Reject H_0 → {flag}
"""
            if 3 in stages:
                cm_result = sms.CompareMeans(
                    sms.DescrStatsW(a), sms.DescrStatsW(b))
                conf_table = cm_result.summary(usevar='pooled', alpha=alpha)
                conf_interval = list(
                    map(float, conf_table.as_text().split('\n')[4].split()[6:]))
                # record result
                result_dict['conf_interval'] = conf_interval
                con_coef = 1 - alpha
                results += f"""
        3. Confidence Interval      
==============================
{con_coef * 100:.1f}% Confidence Interval: [{conf_interval[0]:.4f}, {conf_interval[1]:.4f}]
"""

        if show_table == True and 3 in stages:
            results += f"""{conf_table.as_text()}"""

    if show == True:
        print(results)
    return result_dict
