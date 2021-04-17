from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import scipy.stats as stats

from . import t
from . import chi2
from . import p
from . import ind
from . import anova
from . import non


"""
Using z statistic for hypothesis testing and confidence intervals.

+ rejection_region_method(x_mean, mu, std, n, alpha, option='left', precision=4, show=True, ignore=False)
+ testing_statistic_method
+ p_value_method
+ power_test
+ sample_size
+ type2_plot
+ power_plot
"""


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
    """
    Input: x_mean, mu, std, n, alpha, option='left', precision=4, ignore=False
    Output: 
        if opt == 't':
            return z, z_l, z_u
        else:
            return z, z_value
    """
    opt = option.lower()[0]
    z = (x_mean - mu)/(std / math.sqrt(n))
    if opt == 't':
        option = 'Two-Tail Test'
        z_value = stats.norm.ppf(1 - alpha / 2)
        z_u = z_value
        z_l = -z_value
        flag = z < z_l or z > z_u

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
            flag = z < z_value
        elif opt == 'r':
            option = 'One-Tail Test (right tail)'
            z_value = stats.norm.ppf(1 - alpha)
            flag = z > z_value

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
    """
    Input: x_mean, h0_mean, h0_std (standard deviation of population), samp_num (sample size), siglevel (alpha), option='left', precision=4):
    Output: zcv, p_value
    """
    z_value = (x_mean - h0_mean) / (h0_std/(samp_num ** 0.5))
    alpha = siglevel
    opt = option.lower()[0]
    if opt == 't':
        # two-tail test
        option = 'Two-Tail Test'
        p_value = (1 - stats.norm.cdf(z_value)) * 2
        if (p_value > 1):
            p_value = (stats.norm.cdf(z_value)) * 2
        zcv = stats.norm.ppf(1 - siglevel/2)
        flag = p_value < alpha
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
        flag = p_value < alpha
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


def power_test(x_mean, h0_mean, std, n, alpha, h1_mean, option='left', precision=4, show=True, ignore=True):
    """
    Input: x_mean (not necessary if ignore=True), h0_mean, std, n, alpha, h1_mean, option='left', precision=4, show=True, ignore=True
    Output: type2_p (beta), ptest (power of a test)
    """
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
        # if x_c > h1_mean:
        #     opt = 'l'
        # else:
        #     opt = 'r'

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

    if show:
        print(result)

    return type2_p, ptest


def sample_size(h0_mean, h1_mean, std, alpha, beta):
    """
    Input: h0_mean, h1_mean, std (population), alpha, beta
    Output: desired sample size
    """
    z_a = stats.norm.ppf(1 - alpha)
    z_b = stats.norm.ppf(1 - beta)
    n = (((z_a + z_b) * (std))**2) / ((h0_mean - h1_mean) ** 2)
    return n


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
    if opt == 'r':
        zcv = stats.norm.ppf(1-alpha)
    elif opt == 'l':
        zcv = -stats.norm.ppf(1-alpha)
    elif opt == 't':
        zcv = stats.norm.ppf(1 - alpha / 2)

    means = np.arange(ranges[0], ranges[1], 0.1)
    betas = np.zeros(means.shape[0])
    powers = betas.copy()

    if show:
        fig, ax = plt.subplots(figsize=figsize)

    for nsize in nsizes:
        means = np.arange(ranges[0], ranges[1], 0.1)
        betas = np.zeros(means.shape[0])
        powers = betas.copy()
        i = 0
        if opt == 'r':
            x_c = h0_mean + zcv * psigma / (nsize ** 0.5)
            for h1_mean in means:
                z_type2 = (x_c - h1_mean) / (psigma / (nsize ** 0.5))
                type2_p = stats.norm.cdf(z_type2)
                betas[i] = type2_p
                powers[i] = 1 - type2_p
                i += 1
        elif opt == 'l':
            x_c = h0_mean + zcv * psigma / (nsize ** 0.5)
            for h1_mean in means:
                z_type2 = (x_c - h1_mean) / (psigma / (nsize ** 0.5))
                type2_p = 1 - stats.norm.cdf(z_type2)
                betas[i] = type2_p
                powers[i] = 1 - type2_p
                i += 1
        elif opt == 't':
            x_u = h0_mean + zcv * psigma / math.sqrt(nsize)
            x_l = h0_mean - zcv * psigma / math.sqrt(nsize)
            # x_l, x_u = rejection_region_method(_, h0_mean, psigma, nsize, alpha, option=opt, precision=4, show=False, ignore=True)
            for h1_mean in means:
                z_type2_l = (x_l - h1_mean) / (psigma / (nsize ** 0.5))
                z_type2_u = (x_u - h1_mean) / (psigma / (nsize ** 0.5))
                type2_p_l = stats.norm.cdf(z_type2_l)
                type2_p_u = stats.norm.cdf(z_type2_u)
                type2_p = type2_p_u - type2_p_l
                betas[i] = type2_p
                powers[i] = 1 - type2_p
                i += 1

        if pf:
            if show:
                plt.plot(means, betas, label=f'OC ({nsize})')
                plt.plot(means, powers, label=f'PF ({nsize})')
        else:
            if show:
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


# 2021-04-18

def _single_pop(url=None):
    quit_signal = ['quit', 'q', 'Q', 'Quit', 'QUIT', 'exit']
    print('Describe a single population:')
    cmd = (input(f'''Data Type?
1. Interval
2. Nominal
'''))
    while(cmd not in quit_signal):
        try:
            cmd = int(cmd)
        except:
            pass
        if cmd == 1:
            cmd = (input(f'''Type of descriptive measurements?
1. Central location
2. Variability
'''))
            try:
                cmd = int(cmd)
            except:
                pass
            dic = {1: 't-test and estimator of mu',
                   2: 'chi2-test and estimator of sigma2'}
            print(dic[cmd])
        elif cmd == 2:
            cmd = (input(f'''Number of categories?
1. Two
2. Two or more
'''))
            try:
                cmd = int(cmd)
            except:
                pass
            dic = {1: 'z-test and estimator of p',
                   2: 'chi2-goodness-of-fit test'}
            print(dic[cmd])

        break


def _experimental_design(url=None):
    quit_signal = ['quit', 'q', 'Q', 'Quit', 'QUIT', 'exit']
    cmd = (input(f'''Experimental design?
1. Independent samples
2. Matched Pairs
'''))
    while(cmd not in quit_signal):
        try:
            cmd = int(cmd)
        except:
            pass
        if cmd == 1:
            cmd = (input(f'''Population distributions?
1. Normal
2. Nonnormal
'''))
            try:
                cmd = int(cmd)
            except:
                pass
            dic = {1: _variance, 2: 'Wilcoxon Rank Sum Test'}
            if callable(dic[cmd]):
                dic[cmd](url=url)
            else:
                print(dic[cmd])
            # if type(dic[cmd]) == str:
            #     print(dic[cmd])
        elif cmd == 2:
            cmd = (input(f'''Distribution of differences?
1. Normal
2. Nonnormal
'''))
            try:
                cmd = int(cmd)
            except:
                pass
            dic = {1: 't-test and estimator of \mu_D',
                   2: 'Wilcoxon Signed Rank Sum Test'}
            print(dic[cmd])

        break


def _variance(url=None):
    quit_signal = ['quit', 'q', 'Q', 'Quit', 'QUIT', 'exit']
    cmd = (input(f'''Population variances?
1. Equal
2. Unequal
'''))
    while(cmd not in quit_signal):
        try:
            cmd = int(cmd)
        except:
            pass
        dic = {1: f't-test and estimator of \mu_1 - \mu_2 (equal-variances) ({url}/MGT2002/Chap-13-Inference-about-Comparing-Two-Population/#python-code-and-interpretation)',
               2: f't-test and estimator of \mu_1 - \mu_2 (unequal-variances) ({url}/MGT2002/Chap-13-Inference-about-Comparing-Two-Population/#python-code-and-interpretation)'}
        print(dic[cmd])
        break


def _two_pop(url=None):
    quit_signal = ['quit', 'q', 'Q', 'Quit', 'QUIT', 'exit']
    cmd = (input(f'''Data Type?
1. Interval
2. Ordinal
3. Nominal
'''))
    while(cmd not in quit_signal):
        try:
            cmd = int(cmd)
        except:
            pass
        if cmd == 1:
            cmd = (input(f'''Type of descriptive measurements?
1. Central location
2. Variability
'''))
            try:
                cmd = int(cmd)
            except:
                pass
            dic = {1: _experimental_design,
                   2: 'chi2-test and estimator of sigma2'}
            if callable(dic[cmd]):
                dic[cmd](url=url)
            else:
                print(dic[cmd])
            # if type(dic[cmd]) == str:
            #     print(dic[cmd])
        elif cmd == 2:
            cmd = (input(f'''Experimental Design?
1. Independent samples
2. Matched pairs
'''))
            try:
                cmd = int(cmd)
            except:
                pass
            dic = {1: 'Wilcoxon Rank Sum Test', 2: 'Sign Test'}
            print(dic[cmd])
        elif cmd == 3:
            cmd = (input(f'''Number of categories?
1. Two
2. Two or more
'''))
            try:
                cmd = int(cmd)
            except:
                pass
            dic = {
                1: f'z-test and estimator of p_1 - p_2 ({url}/MGT2002/Chap-13-Inference-about-Comparing-Two-Population/#python-code-and-interpretation_2)', 2: 'chi2-test of a contingency table'}
            print(dic[cmd])

        break


def _pop_dist(url=None):
    quit_signal = ['quit', 'q', 'Q', 'Quit', 'QUIT', 'exit']
    cmd = (input(f'''Population distributions?
1. Normal
2. Nonnormal
'''))
    while(cmd not in quit_signal):
        try:
            cmd = int(cmd)
        except:
            pass
        dic = {1: 'One-way and two-factor analysis of variance',
               2: 'Kruskal-Wallis Test'}
        print(dic[cmd])
        break


def _pop_dist_2(url=None):
    quit_signal = ['quit', 'q', 'Q', 'Quit', 'QUIT', 'exit']
    cmd = (input(f'''Population distributions?
1. Normal
2. Nonnormal
'''))
    while(cmd not in quit_signal):
        try:
            cmd = int(cmd)
        except:
            pass
        dic = {
            1: f'Randomized block analysis of variance ({url}/MGT2002/Chap-14-II-Analysis-of-Variance-ANOVA/#randomized-block-anova-test)', 2: 'Friedman Test'}
        print(dic[cmd])
        break


def _two_or_more_pop(url=None):
    quit_signal = ['quit', 'q', 'Q', 'Quit', 'QUIT', 'exit']
    cmd = (input(f'''Data Type?
1. Interval
2. Ordinal
3. Nominal
'''))
    while(cmd not in quit_signal):
        try:
            cmd = int(cmd)
        except:
            pass
        if cmd == 1:
            cmd = (input(f'''Experimental design?
1. Independent samples
2. Blocks
'''))
            try:
                cmd = int(cmd)
            except:
                pass
            dic = {1: _pop_dist, 2: _pop_dist_2}
            if callable(dic[cmd]):
                dic[cmd](url=url)
            else:
                print(dic[cmd])

        elif cmd == 2:
            cmd = (input(f'''Experimental Design?
1. Independent samples
2. Blocks
'''))
            try:
                cmd = int(cmd)
            except:
                pass
            dic = {1: 'Kruskal-Wallis Test', 2: 'Friedman Test'}
            print(dic[cmd])
        elif cmd == 3:
            print(
                f'chi2-test of a contingency table ({url}/MGT2002/Chap-15-Chi-Squared-Tests/#python-code-for-contingency-test)')

        break


def _relationship(url=None):
    quit_signal = ['quit', 'q', 'Q', 'Quit', 'QUIT', 'exit']
    cmd = (input(f'''Data Type?
1. Interval
2. Ordinal
3. Nominal
'''))
    while(cmd not in quit_signal):
        try:
            cmd = int(cmd)
        except:
            pass
        if cmd == 1:
            cmd = (input(f'''Population distributions?
1. Error is normal or x and y bivariate normal
2. x and y not bivariate normal
'''))
            try:
                cmd = int(cmd)
            except:
                pass
            dic = {1: 'Simple linear regression and correlation',
                   2: 'Spearman rank correlation'}
            if type(dic[cmd]) == str:
                print(dic[cmd])
        elif cmd == 2:
            print('Spearman rank correlation')
        elif cmd == 3:
            print('chi2-test of a contingency table')

        break


def which(location='local'):
    url = ''
    if (location == 'local'):
        url += 'http://127.0.0.1:8000'
    else:
        url = location
    quit_signal = ['quit', 'q', 'Q', 'Quit', 'QUIT', 'exit']
    cmd = input(f'''Declare objective:
1. Describe a single population
2. Comapre two populations
3. Compare two or more populations
4. Analyze relationships between two variables

(type 'quit' to quit the program)
''')
    while(cmd not in quit_signal):
        try:
            cmd = int(cmd)
        except:
            pass
        if cmd == 1:
            _single_pop(url=url)
        elif cmd == 2:
            _two_pop(url=url)
        elif cmd == 3:
            _two_or_more_pop(url=url)
        elif cmd == 4:
            _relationship(url=url)

        cmd = input(f'''=======
Declare objective:
1. Describe a single population
2. Comapre two populations
3. Compare two or more populations
4. Analyze relationships between two variables

(type 'quit' to quit the program)
''')
