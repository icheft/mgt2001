from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import scipy.stats as stats


def con_level(x_bar, sigma, n, sig_level, show=True):
    a = sig_level
    df_v = n - 1
    con_coef = 1 - a
    t_value = stats.t.ppf(1 - a / 2, df=df_v)
    sig_x_bar = sigma / math.sqrt(n)
    lcl = x_bar - t_value * sig_x_bar
    ucl = x_bar + t_value * sig_x_bar
    result = f"""{con_coef * 100:.1f}% Confidence Interval: [{lcl:.4f}, {ucl:.4f}]
Mean: {x_bar:.4f}
Std. Dev. = {sigma:.4f}
Sample Size: {n}
t (Critical value): {t_value:.4f}
    """
    if show:
        print(result)
    return {"lcl": lcl, "ucl": ucl, "x_bar": x_bar, "t_value": t_value, "sig_x_bar": sig_x_bar}


def rejection_region_method(x_mean, mu, std, n, alpha, option='left', precision=4, show=True, ignore=False):
    opt = option.lower()[0]
    df_v = n - 1
    if opt == 't':
        option = 'Two-Tail Test'
        t_value = stats.t.ppf(1 - alpha / 2, df=df_v)
        x_u = mu + t_value * std / math.sqrt(n)
        x_l = mu - t_value * std / math.sqrt(n)
        flag = x_mean < x_l or x_mean > x_u
        if not ignore:
            result = f'''======= The Rejection Region Method =======
Significant Level (alpha) = {alpha:.{precision}f}
t (Critical value) = {t_value:.{precision}f}

Using {option}:
x̄ =  {x_mean:.{precision}f}
x_l (Lower bound for the critical value) = {x_l:.{precision}f}
x_u (Upper bound for the critical value) = {x_u:.{precision}f}
Reject H_0 → {flag}
            '''
        else:
            result = f'''======= The Rejection Region Method =======
Significant Level (alpha) = {alpha:.{precision}f}
t (Critical value) = {t_value:.{precision}f}

Using {option}:
x_l (Lower bound for the critical value) = {x_l:.{precision}f}
x_u (Upper bound for the critical value) = {x_u:.{precision}f}
            '''

    else:
        if opt == 'l':
            # left tail
            option = 'One-Tail Test (left tail)'
            t_value = stats.t.ppf(alpha, df=df_v)  # negative
            x_c = mu + t_value * std / math.sqrt(n)
            flag = x_mean < x_c
        elif opt == 'r':
            option = 'One-Tail Test (right tail)'
            t_value = stats.t.ppf(1 - alpha, df=df_v)
            x_c = mu + t_value * std / math.sqrt(n)
            flag = x_mean > x_c
        if not ignore:
            result = f'''======= The Rejection Region Method =======
Significant Level (alpha) = {alpha:.{precision}f}
t (Critical value) = {t_value:.{precision}f}

Using {option}:
x̄ =  {x_mean:.{precision}f}
x_c (Critical value) = {x_c:.{precision}f}
Reject H_0 → {flag}
            '''
        else:
            result = f'''======= The Rejection Region Method =======
Significant Level (alpha) = {alpha:.{precision}f}
t (Critical value) = {t_value:.{precision}f}

Using {option}:
x_c (Critical value) = {x_c:.{precision}f}
            '''

    if show:
        print(result)

    if opt == 't':
        return x_l, x_u
    else:
        return x_c


def testing_statistic_method(x_mean, mu, std, n, alpha, option='left', precision=4, ignore=False):
    df_v = n - 1
    opt = option.lower()[0]
    t = (x_mean - mu)/(std / math.sqrt(n))
    if opt == 't':
        option = 'Two-Tail Test'
        t_value = stats.t.ppf(1 - alpha / 2, df=df_v)
        t_u = t_value
        t_l = -t_value
        flag = t <= t_l or t >= t_u

        if not ignore:
            result = f'''======= Testing Statistic Method =======
Significant Level (alpha) = {alpha:.{precision}f}
t (Critical value) = {t_value:.{precision}f}

Using {option}:
t (Observed value) =  {t:.{precision}f}
t_l (Lower bound for the critical value) = {t_l:.{precision}f}
t_u (Upper bound for the critical value) = {t_u:.{precision}f}
Reject H_0 → {flag}
            '''
        else:
            result = f'''======= Testing Statistic Method =======
Significant Level (alpha) = {alpha:.{precision}f}
t (Critical value) = {t_value:.{precision}f}

Using {option}:
t_l (Lower bound for the critical value) = {t_l:.{precision}f}
t_u (Upper bound for the critical value) = {t_u:.{precision}f}
            '''

    else:
        if opt == 'l':
            # left tail
            option = 'One-Tail Test (left tail)'
            t_value = stats.t.ppf(alpha, df=df_v)  # negative
            flag = t <= t_value
        elif opt == 'r':
            option = 'One-Tail Test (right tail)'
            t_value = stats.t.ppf(1 - alpha, df=df_v)
            flag = t >= t_value

        if not ignore:
            result = f'''======= Testing Statistic Method =======
Significant Level (alpha) = {alpha:.{precision}f}
t (Critical value) = {t_value:.{precision}f}

Using {option}:
t (Observed value) =  {t:.{precision}f}
t (Critical value) = {t_value:.{precision}f}
Reject H_0 → {flag}
            '''

        else:
            result = f'''======= Testing Statistic Method =======
Significant Level (alpha) = {alpha:.{precision}f}
t (Critical value) = {t_value:.{precision}f}

Using {option}:
t (Critical value) = {t_value:.{precision}f}
            '''

    print(result)
    if opt == 't':
        return t, t_l, t_u
    else:
        return t, t_value


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


def p_value_method(x_mean, h0_mean, std, samp_num, siglevel, option='left', precision=4):
    df_v = samp_num - 1
    t_value = (x_mean - h0_mean) / (std/(samp_num ** 0.5))
    alpha = siglevel
    opt = option.lower()[0]
    if opt == 't':
        # two-tail test
        option = 'Two-Tail Test'
        p_value = (1 - stats.t.cdf(t_value, df=df_v)) * 2
        tcv = stats.t.ppf(1 - siglevel/2, df=df_v)
        flag = p_value <= alpha
        sub_result = f'''Using {option}:
Difference = {x_mean - h0_mean}
t (Critical value) = {-tcv:.{precision}f}, {tcv:.{precision}f}
t (Observed value) = {t_value:.{precision}f}
p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})
Reject H_0 → {flag}
        '''
    else:
        if opt == 'l':
            option = 'One-Tail Test (left tail)'
            p_value = stats.t.cdf(t_value, df=df_v)
            tcv = stats.t.ppf(siglevel, df=df_v)
        elif opt == 'r':
            option = 'One-Tail Test (right tail)'
            p_value = stats.t.sf(t_value, df=df_v)
            tcv = stats.t.ppf(1 - siglevel, df=df_v)
        flag = p_value <= alpha
        sub_result = f'''Using {option}:
Difference = {x_mean - h0_mean}
t (Critical value) = {tcv:.{precision}f}
t (Observed value) = {t_value:.{precision}f}
p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})
Reject H_0 → {flag}
        '''

    result = f"""======= p-value Method =======
Mean = {x_mean:.{precision}f}
Number of Observation = {samp_num:.{precision}f}
Hypothesized Mean (H0 Mean) = {h0_mean:.{precision}f}
Sample Standard Deviation = {std:.{precision}f}
Significant Level (alpha) = {siglevel:.{precision}f}

""" + sub_result

    print(result)

    return tcv, p_value
