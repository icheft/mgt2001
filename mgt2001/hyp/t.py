from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import scipy.stats as stats


def con_level(x_bar, sigma, n, sig_level, show=True):
    """
Input: x_bar (x_mean), sigma (sample sigma), sample size, sig_level (alpha), show=True
Return the confidence level at alpha. Return a dictionary: {"lcl": lcl, "ucl": ucl, "x_bar": x_bar, "t_value": t_value, "sig_x_bar": sig_x_bar}

+ `show`: default is `True`. Set to `False` to disable rendering.
    """
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
    """
    Input: x_mean, mu, std, n, alpha, option='left', precision=4, show=True, ignore=False
    Output: 
        if opt == 't':
            return x_l, x_u
        else:
            return x_c
    """
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
    """
    Input: x_mean, mu, std (sample std), n, alpha, option='left', precision=4, ignore=False
    Output: 
        if opt == 't':
            return t, t_l, t_u
        else:
            return t, t_value
    """
    df_v = n - 1
    opt = option.lower()[0]
    t = (x_mean - mu)/(std / math.sqrt(n))
    if opt == 't':
        option = 'Two-Tail Test'
        t_value = stats.t.ppf(1 - alpha / 2, df=df_v)
        t_u = t_value
        t_l = -t_value
        flag = t < t_l or t > t_u

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
            flag = t < t_value
        elif opt == 'r':
            option = 'One-Tail Test (right tail)'
            t_value = stats.t.ppf(1 - alpha, df=df_v)
            flag = t > t_value

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
    """
    Input: x_mean, h0_mean, std (standard deviation of sample), samp_num (sample size), siglevel (alpha), option='left', precision=4):
    Output: zcv, p_value
    """
    df_v = samp_num - 1
    t_value = (x_mean - h0_mean) / (std/(samp_num ** 0.5))
    alpha = siglevel
    opt = option.lower()[0]
    if opt == 't':
        # two-tail test
        option = 'Two-Tail Test'
        p_value = (1 - stats.t.cdf(t_value, df=df_v)) * 2
        if (p_value > 1):
            p_value = (stats.t.cdf(t_value, df=df_v)) * 2
        tcv = stats.t.ppf(1 - siglevel/2, df=df_v)
        flag = p_value < alpha
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
        flag = p_value < alpha
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


def type2_plot(h0_mean, psigma, nsizes, alpha, ranges, option='right', figsize=(12, 6), pf=False, label=True, show=True):
    """
    Caution: 外面要自己 plt.show()

    Input: h0_mean, psigma, nsizes (list or one value), alpha, ranges, option='right', figsize=(12, 6), pf=False, label=True, show=True
    → set show to false to only get the values for powers
    Output: (if pf=True: means, betas, xticks, yticks)
    """

    try:
        _ = iter(nsizes)
    except TypeError as te:
        nsizes = [nsizes]

    opt = option.lower()[0]
    # options

    means = np.arange(ranges[0], ranges[1], 0.1)
    betas = np.zeros(means.shape[0])
    powers = betas.copy()
    if show:
        fig, ax = plt.subplots(figsize=figsize)

    for nsize in nsizes:
        df_v = nsize - 1
        if opt == 'r':
            tcv = stats.t.ppf(1 - alpha, df=df_v)
        elif opt == 'l':
            tcv = stats.t.ppf(alpha, df=df_v)
        elif opt == 't':
            tcv = stats.t.ppf(1 - alpha / 2, df=df_v)
        means = np.arange(ranges[0], ranges[1], 0.1)
        betas = np.zeros(means.shape[0])
        powers = betas.copy()
        i = 0
        if opt == 'r':
            x_c = h0_mean + tcv * psigma / (nsize ** 0.5)
            for h1_mean in means:
                t_type2 = (x_c - h1_mean) / (psigma / (nsize ** 0.5))
                type2_p = stats.t.cdf(t_type2, df=df_v)
                betas[i] = type2_p
                powers[i] = 1 - type2_p
                i += 1
        elif opt == 'l':
            x_c = h0_mean + tcv * psigma / (nsize ** 0.5)
            for h1_mean in means:
                t_type2 = (x_c - h1_mean) / (psigma / (nsize ** 0.5))
                type2_p = 1 - stats.t.cdf(t_type2, df=df_v)
                betas[i] = type2_p
                powers[i] = 1 - type2_p
                i += 1
        elif opt == 't':
            x_u = h0_mean + tcv * psigma / math.sqrt(nsize)
            x_l = h0_mean - tcv * psigma / math.sqrt(nsize)
            # x_l, x_u = rejection_region_method(_, h0_mean, psigma, nsize, alpha, option=opt, precision=4, show=False, ignore=True)
            for h1_mean in means:
                t_type2_l = (x_l - h1_mean) / (psigma / (nsize ** 0.5))
                t_type2_u = (x_u - h1_mean) / (psigma / (nsize ** 0.5))
                type2_p_l = stats.t.cdf(t_type2_l, df=df_v)
                type2_p_u = stats.t.cdf(t_type2_u, df=df_v)
                type2_p = type2_p_u - type2_p_l
                betas[i] = type2_p
                powers[i] = 1 - type2_p
                i += 1

        if show:
            if pf:
                plt.plot(means, betas, label=f'OC ({nsize})')
                plt.plot(means, powers, label=f'PF ({nsize})')
            else:
                plt.plot(means, betas, label=f'n = {nsize}')

    if len(ranges) == 3:
        xticks = np.arange(ranges[0], ranges[1] + 1, ranges[2])
    else:  # default
        xticks = np.arange(ranges[0], ranges[1] + 1, 1)
    yticks = np.arange(0, 1.1, .1)

    if show:
        plt.xlabel("H1 Mean")
        plt.xticks(xticks, rotation=45, fontsize=8)
        plt.yticks(yticks, fontsize=8)
        plt.ylabel("Probability of a Type II Error")
        plt.margins(x=.01, tight=False)
        if label:
            plt.legend()

    if pf:
        return means, betas, xticks, yticks


def power_test(x_mean, h0_mean, std, n, alpha, h1_mean, option='left', precision=4, show=True, ignore=True):
    """
    Input: x_mean (not necessary if ignore=True), h0_mean, std, n, alpha, h1_mean, option='left', precision=4, show=True, ignore=True
    Output: type2_p (beta), ptest (power of a test)
    """
    opt = option.lower()[0]
    df_v = (n - 1)
    if opt == 't':
        option = 'Two-Tail Test'
        x_l, x_u = rejection_region_method(
            x_mean, h0_mean, std, n, alpha, option=opt, precision=precision, show=show, ignore=ignore)
        t_value = stats.t.ppf(1 - alpha / 2, df=df_v)
        t_l = -t_value
        t_u = t_value
        t_type2_l = (x_l - h1_mean) / (std / (n ** 0.5))
        t_type2_u = (x_u - h1_mean) / (std / (n ** 0.5))
        type2_p_l = stats.t.cdf(t_type2_l, df=df_v)
        type2_p_u = stats.t.cdf(t_type2_u, df=df_v)
        type2_p = type2_p_u - type2_p_l
        ptest = 1 - type2_p
        result = f'''======= Evaluating Type II Errors ({option}) =======
μ = {h1_mean}
t (lower bound) = {t_type2_l:.{precision}f}
t (upper bound) = {t_type2_u:.{precision}f}

t_l (Lower bound for the critical value) = {t_l:.{precision}f}
t_u (Upper bound for the critical value) = {t_u:.{precision}f}

x_l (Lower bound for x critical value) = {x_l:.{precision}f}
x_u (Upper bound for x critical value) = {x_u:.{precision}f}

P(Type II Error) = {type2_p:.{precision}f}
Power of a Test = {ptest:.{precision}f}
        '''
    else:
        x_c = rejection_region_method(
            x_mean, h0_mean, std, n, alpha, option=opt, precision=precision, show=show, ignore=ignore)
#         if x_c > h1_mean:
#             opt = 'l'
#         else:
#             opt = 'r'

        if opt == 'l':
            option = 'One-Tail Test (left tail)'

            t_c = stats.t.ppf(alpha, df=df_v)
            t_type2 = (x_c - h1_mean) / (std / (n ** 0.5))
            type2_p = 1 - stats.t.cdf(t_type2, df=df_v)
            ptest = 1 - type2_p

        elif opt == 'r':
            option = 'One-Tail Test (right tail)'
            t_c = stats.t.ppf(1 - alpha, df=df_v)
            t_type2 = (x_c - h1_mean) / (std / (n ** 0.5))
            type2_p = stats.t.cdf(t_type2, df=df_v)
            ptest = 1 - type2_p

        result = f'''======= Evaluating Type II Errors ({option}) =======
μ = {h1_mean}
t = {t_type2:.{precision}f}

t critical value = {t_c:.{precision}f}
x critical value = {x_c:.{precision}f}

P(Type II Error) = {type2_p:.{precision}f}
Power of a Test = {ptest:.{precision}f}
'''

    if show:
        print(result)

    return type2_p, ptest


def power_plot(h0_mean, psigma, nsizes, alpha, ranges, option='r', figsize=(12, 6), show=True):
    means, betas, xticks, yticks = type2_plot(
        h0_mean, psigma, nsizes, alpha, ranges, option=option, figsize=figsize, pf=True, label=True, show=show)
    if show:
        plt.clf()
        plt.plot(means, 1 - betas)
        plt.xticks(xticks, rotation=45, fontsize=8)
        plt.yticks(yticks, fontsize=8)
        plt.title('Power Function Curve')
        plt.margins(x=.01, tight=False)
