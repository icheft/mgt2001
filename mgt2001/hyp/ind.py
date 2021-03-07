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


def two_population(a, b, alpha=.05, consistency='equal', option='right', show_table=False, stages=[1, 2, 3], show=True, precision=4, matched_pairs=False):
    """
+ [First stage]: F Statistics - consistency: equal, left (1 is more consistent than 2), right (2 is more consistent than 1)
+ [Second stage]: t Test
+ [Third stage]: Confidence Interval

Will return a result_dict regardless of stages.
    """
    opt = option.lower()[0]
    results = ""

    const = consistency.lower()[0]

    result_dict = dict()

    df_1 = len(a) - 1
    df_2 = len(b) - 1
    if 1 in stages:

        varall = [stats.describe(a).variance,
                  stats.describe(b).variance]
        f_value = varall[0] / varall[1]

        result_dict['varall'] = varall
        result_dict['f_value'] = f_value

        ptmp = stats.f.cdf(f_value, df_1, df_2)

        if const == 'e':
            if ptmp > 0.5:
                ptmp = 1 - ptmp
            p_value = ptmp * 2
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
                p_value = 1 - ptmp
                if f_value > rej_upper:
                    flag = True
                else:
                    flag = False
                text = 'σ_1/σ_2 > 1'
            else:
                result_dict['f_rej_lower'] = rej_lower
                p_value = ptmp
                if f_value < rej_lower:
                    flag = True
                else:
                    flag = False
                text = 'σ_1/σ_2 < 1'

        result_dict['p_value'] = p_value

        results = f"""          F Statistics
===================================
F statistic = {f_value:.{precision}f}
p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})
Reject H_0 ({text}) → {flag}
"""
    if 2 in stages:
        if matched_pairs:
            samp_diff = a - b
            nobs = samp_diff.shape[0]
            df = nobs - 1

            tmpdesc = stats.describe(samp_diff)
            t_value = tmpdesc.mean / (tmpdesc.variance ** 0.5) * (nobs ** 0.5)

            # p-values
            ptmp = stats.t.cdf(t_value, df)
            if opt == 'r':
                text = 'one-tail'
                tcv = stats.t.ppf(1 - alpha, df=df)
                p_value = 1 - ptmp
            elif opt == 'l':
                text = 'one-tail'
                p_value = ptmp
                tcv = stats.t.ppf(alpha, df=df)
            else:
                text = 'two-tail'
                tcv = stats.t.ppf(1 - alpha/2, df=df)
                if ptmp > 0.5:
                    ptmp = 1 - ptmp
                p_value = ptmp * 2

            flag = p_value < alpha
            results += f"""
           t Test      
===================================
t (Observed value) = {t_value:.{precision}f}
p-value ({text}) = {p_value:.{precision}f} ({inter_p_value(p_value)})
t (Critical, ({text})) = {tcv:.{precision}f}
DF = {(df):.{precision}f}
Reject H_0 → {flag}
"""
            result_dict['t_p_value'] = p_value
            result_dict['t_critical_value'] = tcv
            result_dict['t_observed_value'] = t_value
            t_alpha = stats.t.ppf(1 - alpha / 2, df)
            std_xbar = (tmpdesc.variance / nobs) ** 0.5
            LCL = tmpdesc.mean - t_alpha * std_xbar
            UCL = tmpdesc.mean + t_alpha * std_xbar
            con_coef = 1 - alpha
            conf_interval = [LCL, UCL]
            result_dict['conf_interval'] = conf_interval
            results += f"""
           Confidence Interval      
===================================
{con_coef * 100:.1f}% Confidence Interval: [{LCL:.{precision}f}, {UCL:.{precision}f}]
"""
        else:
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
           t Test      
===================================
t (Observed value) = {t_summary[0]:.{precision}f}
p-value (two-tail) = {t_summary[1]:.{precision}f} ({inter_p_value(t_summary[1])})
t (Critical, two-tail) = {t_critical_two:.{precision}f}
DF = {(df_1 + df_2):.{precision}f}
Reject H_0 → {flag}
"""
                else:
                    flag = t_summary[1] / 2 < alpha
                    result_dict['t_observed_value'] = t_summary[0]
                    result_dict['t_p_value'] = t_summary[1] / 2
                    result_dict['df'] = df_1 + df_2
                    results += f"""
           t Test      
===================================
t (Observed value) = {t_summary[0]:.{precision}f}
p-value (one-tail) = {(t_summary[1] / 2):.{precision}f} ({inter_p_value(t_summary[1] / 2)})
t (Critical, one-tail) = {t_critical_one:.{precision}f}
DF = {(df_1 + df_2):.{precision}f}
Reject H_0 → {flag}
"""
                if 3 in stages:
                    cm_result = sms.CompareMeans(
                        sms.DescrStatsW(a), sms.DescrStatsW(b))
                    conf_table = cm_result.summary(
                        usevar='unequal', alpha=alpha)
                    conf_interval = list(
                        map(float, conf_table.as_text().split('\n')[4].split()[6:]))
                    con_coef = 1 - alpha

                    # record result
                    result_dict['conf_interval'] = conf_interval
                    results += f"""
           Confidence Interval      
===================================
{con_coef * 100:.1f}% Confidence Interval: [{conf_interval[0]:.{precision}f}, {conf_interval[1]:.{precision}f}]
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
           t Test      
===================================
t (Observed value) = {t_summary[0]:.{precision}f}
p-value (two-tail) = {t_summary[1]:.{precision}f} ({inter_p_value(t_summary[1])})
t (Critical, two-tail) = {t_critical_two:.{precision}f}
DF = {(df_1 + df_2):.{precision}f}
Reject H_0 → {flag}
"""
                else:
                    flag = t_summary[1] / 2 < alpha
                    result_dict['t_observed_value'] = t_summary[0]
                    result_dict['t_p_value'] = t_summary[1]
                    result_dict['df'] = df_1 + df_2

                    results += f"""
           t Test      
===================================
t (Observed value) = {t_summary[0]:.{precision}f}
p-value (one-tail) = {(t_summary[1] / 2):.{precision}f} ({inter_p_value(t_summary[1] / 2)})
t (Critical, one-tail) = {t_critical_one:.{precision}f}
DF = {(df_1 + df_2):.{precision}f}
Reject H_0 → {flag}
"""
                if 3 in stages:
                    cm_result = sms.CompareMeans(
                        sms.DescrStatsW(a), sms.DescrStatsW(b))
                    conf_table = cm_result.summary(
                        usevar='pooled', alpha=alpha)
                    conf_interval = list(
                        map(float, conf_table.as_text().split('\n')[4].split()[6:]))
                    # record result
                    result_dict['conf_interval'] = conf_interval
                    con_coef = 1 - alpha
                    results += f"""
           Confidence Interval      
===================================
{con_coef * 100:.1f}% Confidence Interval: [{conf_interval[0]:.{precision}f}, {conf_interval[1]:.{precision}f}]
"""

            if show_table == True and 3 in stages:
                results += f"""{conf_table.as_text()}"""

    if show == True:
        print(results)
    return result_dict


def _check_normality(n1, n2, p1, p2):
    if n1 * p1 >= 5 and n2 * p2 >= 5 and n1 * (1-p1) >= 5 and n2 * (1-p2) >= 5:
        return True
    else:
        return False


def two_population_proportion(a, b, D, option='right', alpha=0.05, precision=4, show=True):

    opt = option.lower()[0]

    p1 = a.mean()
    p2 = b.mean()
    n1, n2 = len(a), len(b)
    result_dict = dict()
    result_dict['D'] = D
    result_dict['p1'] = p1
    result_dict['p2'] = p2
    result_dict['n1'] = n1
    result_dict['n2'] = n2

    result_dict['Normal'] = _check_normality(n1, n2, p1, p2)

    if D == 0:
        ab_concat = np.concatenate([a, b])
        p_pool = ab_concat.mean()
        sd_p = (p_pool * (1 - p_pool) *
                (1 / n1 + 1 / n2)) ** 0.5
    else:
        sd_p = (p1 * (1-p1) / n1 + p2 * (1 - p2) / n2) ** 0.5

    result_dict['sd_p'] = sd_p
    z_value = ((p1 - p2) - D) / sd_p

    result_dict['z_value'] = z_value

    p_value = 1 - stats.norm.cdf(z_value)  # right

    if opt == 't':
        # two-tail test
        text = 'Two-Tail Test'
        if p_value > 0.5:
            p_value = 1 - p_value
        p_value *= 2

        zcv = stats.norm.ppf(1 - alpha/2)
        flag = p_value < alpha
        sub_result = f'''Using {text}:
z (Observed value, {text}) = {z_value:.{precision}f}
z (Critical value, {text}) = {-zcv:.{precision}f}, {zcv:.{precision}f}
p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})
Reject H_0 → {flag}'''
    else:
        if opt == 'l':
            text = 'One-Tail Test (left tail)'
            p_value = stats.norm.cdf(z_value)
            zcv = -stats.norm.ppf(1 - alpha)
        elif opt == 'r':
            text = 'One-Tail Test (right tail)'
            zcv = stats.norm.ppf(1 - alpha)
        flag = p_value < alpha
        sub_result = f'''Using {text}:
z (Observed value) = {z_value:.{precision}f}
z (Critical value) = {zcv:.{precision}f}
p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})
Reject H_0 → {flag}'''

    result_dict['p_value'] = p_value
    result_dict['zcv'] = zcv

    zcv = stats.norm.ppf(1 - alpha/2)
    con_coef = 1 - alpha
    sd_p = (p1 * (1-p1) / n1 + p2 * (1 - p2) / n2) ** 0.5  # always
    LCL = p1-p2 - zcv*sd_p
    UCL = p1-p2 + zcv*sd_p
    conf_interval = [LCL, UCL]
    result_dict['conf_interval'] = conf_interval

    result = f"""======= Inf. Two Population Proportions =======
D = {D:.{precision}f}
p1 = {p1:.{precision}f}
p2 = {p2:.{precision}f}

""" + sub_result + f"""

{con_coef * 100:.1f}% Confidence Interval: [{LCL:.{precision}f}, {UCL:.{precision}f}]"""

    if show:
        print(result)

    return result_dict
