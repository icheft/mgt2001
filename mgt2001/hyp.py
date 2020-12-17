import pandas as pd
import numpy as np
import math
import scipy.stats as stats


def power_test(x_mean, h0_mean, std, n, alpha, h1_mean, option='left', precision=4, show=True, ignore=True):
    opt = option.lower()[0]
    if opt == 't':
        option = 'Two-Tail Test'
        x_l, x_u = rejection_region_method(
            x_mean, h0_mean, std, n, alpha, option=opt, precision=precision, show=show, ignore=ignore)
        z_value = stats.norm.ppf(1 - alpha / 2)
        z_l = -z_value
        z_u = z_value
        z_type2_l = (x_l - h1_mean) / (std / (n ** 0.5))
        z_type2_u = (x_u - h1_mean) / (std / (n ** 0.5))
        type2_p_l = stats.norm.cdf(z_type2_l)
        type2_p_u = stats.norm.cdf(z_type2_u)
        type2_p = type2_p_u - type2_p_l
        ptest = 1 - type2_p
        result = f'''======= Evaluating Type II Errors ({option}) =======
μ = {h1_mean}
z (lower bound) = {z_type2_l:.{precision}f}
z (upper bound) = {z_type2_u:.{precision}f}

z_l (Lower bound for the critical value) = {z_l:.{precision}f}
z_u (Upper bound for the critical value) = {z_u:.{precision}f}

x_l (Lower bound for x critical value) = {x_l:.{precision}f}
x_u (Upper bound for x critical value) = {x_u:.{precision}f}

P(Type II Error) = {type2_p:.{precision}f}
Power of a Test = {ptest:.{precision}f}
        '''
    else:
        x_c = rejection_region_method(
            x_mean, h0_mean, std, n, alpha, option=opt, precision=precision, show=show, ignore=ignore)
        if opt == 'l':
            option = 'One-Tail Test (left tail)'
            z_c = -stats.norm.ppf(1 - alpha)
            z_type2 = (x_c - h1_mean) / (std / (n ** 0.5))
            type2_p = 1 - stats.norm.cdf(z_type2)
            ptest = 1 - type2_p

        elif opt == 'r':
            option = 'One-Tail Test (right tail)'
            z_c = stats.norm.ppf(1 - alpha)
            z_type2 = (x_c - h1_mean) / (std / (n ** 0.5))
            type2_p = stats.norm.cdf(z_type2)
            ptest = 1 - type2_p

        result = f'''======= Evaluating Type II Errors ({option}) =======
μ = {h1_mean}
z = {z_type2:.{precision}f}

z critical value = {z_c:.{precision}f}
x critical value = {x_c:.{precision}f}

P(Type II Error) = {type2_p:.{precision}f}
Power of a Test = {ptest:.{precision}f}
'''

    print(result)

    return type2_p, ptest


def rejection_region_method(x_mean, mu, std, n, alpha, option='left', precision=4, show=True, ignore=False):
    opt = option.lower()[0]
    if opt == 't':
        option = 'Two-Tail Test'
        z_value = stats.norm.ppf(1 - alpha / 2)
        x_u = mu + z_value * std / math.sqrt(n)
        x_l = mu - z_value * std / math.sqrt(n)
        flag = x_mean < x_l or x_mean > x_u
        if not ignore:
            result = f'''======= The Rejection Region Method =======
Significant Level (alpha) = {alpha:.{precision}f}
z_value = {z_value:.{precision}f}

Using {option}:
x̄ =  {x_mean:.{precision}f}
x_l (Lower bound for the critical value) = {x_l:.{precision}f}
x_u (Upper bound for the critical value) = {x_u:.{precision}f}
Reject H_0 → {flag}
            '''
        else:
            result = f'''======= The Rejection Region Method =======
Significant Level (alpha) = {alpha:.{precision}f}
z_value = {z_value:.{precision}f}

Using {option}:
x_l (Lower bound for the critical value) = {x_l:.{precision}f}
x_u (Upper bound for the critical value) = {x_u:.{precision}f}
            '''

    else:
        if opt == 'l':
            # left tail
            option = 'One-Tail Test (left tail)'
            z_value = stats.norm.ppf(alpha)  # negative
            x_c = mu + z_value * std / math.sqrt(n)
            flag = x_mean < x_c
        elif opt == 'r':
            option = 'One-Tail Test (right tail)'
            z_value = stats.norm.ppf(1 - alpha)
            x_c = mu + z_value * std / math.sqrt(n)
            flag = x_mean > x_c
        if not ignore:
            result = f'''======= The Rejection Region Method =======
Significant Level (alpha) = {alpha:.{precision}f}
z_value = {z_value:.{precision}f}

Using {option}:
x̄ =  {x_mean:.{precision}f}
x_c (Critical value) = {x_c:.{precision}f}
Reject H_0 → {flag}
            '''
        else:
            result = f'''======= The Rejection Region Method =======
Significant Level (alpha) = {alpha:.{precision}f}
z_value = {z_value:.{precision}f}

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
    opt = option.lower()[0]
    z = (x_mean - mu)/(std / math.sqrt(n))
    if opt == 't':
        option = 'Two-Tail Test'
        z_value = stats.norm.ppf(1 - alpha / 2)
        z_u = z_value
        z_l = -z_value
        flag = z <= z_l or z >= z_u

        if not ignore:
            result = f'''======= Testing Statistic Method =======
Significant Level (alpha) = {alpha:.{precision}f}
z_value = {z_value:.{precision}f}

Using {option}:
z =  {z:.{precision}f}
z_l (Lower bound for the critical value) = {z_l:.{precision}f}
z_u (Upper bound for the critical value) = {z_u:.{precision}f}
Reject H_0 → {flag}
            '''
        else:
            result = f'''======= Testing Statistic Method =======
Significant Level (alpha) = {alpha:.{precision}f}
z_value = {z_value:.{precision}f}

Using {option}:
z_l (Lower bound for the critical value) = {z_l:.{precision}f}
z_u (Upper bound for the critical value) = {z_u:.{precision}f}
            '''

    else:
        if opt == 'l':
            # left tail
            option = 'One-Tail Test (left tail)'
            z_value = stats.norm.ppf(alpha)  # negative
            flag = z <= z_value
        elif opt == 'r':
            option = 'One-Tail Test (right tail)'
            z_value = stats.norm.ppf(1 - alpha)
            flag = z >= z_value

        if not ignore:
            result = f'''======= Testing Statistic Method =======
Significant Level (alpha) = {alpha:.{precision}f}
z_value = {z_value:.{precision}f}

Using {option}:
z =  {z:.{precision}f}
z_alpha (Critical value) = {z_value:.{precision}f}
Reject H_0 → {flag}
            '''

        else:
            result = f'''======= Testing Statistic Method =======
Significant Level (alpha) = {alpha:.{precision}f}
z_value = {z_value:.{precision}f}

Using {option}:
z_alpha (Critical value) = {z_value:.{precision}f}
            '''

    print(result)
    if opt == 't':
        return z, z_l, z_u
    else:
        return z, z_value


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


def p_value_method(x_mean, h0_mean, h0_std, samp_num, siglevel, option='left', precision=4):
    z_value = (x_mean - h0_mean) / (h0_std/(samp_num ** 0.5))
    alpha = siglevel
    opt = option.lower()[0]
    if opt == 't':
        # two-tail test
        option = 'Two-Tail Test'
        p_value = (1 - stats.norm.cdf(z_value)) * 2
        zcv = stats.norm.ppf(1 - siglevel/2)
        flag = p_value <= alpha
        sub_result = f'''Using {option}:
Difference = {x_mean - h0_mean}
z (Critical value) = {-zcv:.{precision}f}, {zcv:.{precision}f}
z (Observed value) = {z_value:.{precision}f}
p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})
Reject H_0 → {flag}
        '''
    else:
        if opt == 'l':
            option = 'One-Tail Test (left tail)'
            p_value = stats.norm.cdf(z_value)
            zcv = stats.norm.ppf(siglevel)
        elif opt == 'r':
            option = 'One-Tail Test (right tail)'
            p_value = stats.norm.sf(z_value)
            zcv = stats.norm.ppf(1 - siglevel)
        flag = p_value <= alpha
        sub_result = f'''Using {option}:
Difference = {x_mean - h0_mean}
z (Critical value) = {zcv:.{precision}f}
z (Observed value) = {z_value:.{precision}f}
p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})
Reject H_0 → {flag}
        '''

    result = f"""======= p-value Method =======
Mean = {x_mean:.{precision}f}
Number of Observation = {samp_num:.{precision}f}
Hypothesized Mean (H0 Mean) = {h0_mean:.{precision}f}
Assumed Standard Deviation = {h0_std:.{precision}f}
Significant Level (alpha) = {siglevel:.{precision}f}

""" + sub_result

    print(result)

    return zcv, p_value
