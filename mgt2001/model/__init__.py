from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import math
from io import StringIO
import scipy.stats as stats
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf

_raw_r_table = StringIO(""",5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
2,,,,,,,,"(2, 6)","(2, 6)","(2, 6)","(2, 6)","(2, 6)","(2, 6)","(2, 6)","(2, 6)","(2, 6)"
3,,"(2, 8)","(2, 8)","(2, 8)","(2, 8)","(2, 8)","(2, 8)","(2, 8)","(2, 8)","(2, 8)","(3, 8)","(3, 8)","(3, 8)","(3, 8)","(3, 8)","(3, 8)"
4,"(2, 9)","(2, 9)","(2, 10)","(3, 10)","(3, 10)","(3, 10)","(3, 10)","(3, 10)","(3, 10)","(3, 10)","(3, 10)","(4, 10)","(4, 10)","(4, 10)","(4, 10)","(4, 10)"
5,"(2, 10)","(3, 10)","(3, 11)","(3, 11)","(3, 12)","(3, 12)","(4, 12)","(4, 12)","(4, 12)","(4, 12)","(4, 12)","(4, 12)","(4, 12)","(5, 12)","(5, 12)","(5, 12)"
6,,"(3, 11)","(3, 12)","(3, 12)","(4, 13)","(4, 13)","(4, 13)","(4, 13)","(5, 14)","(5, 14)","(5, 14)","(5, 14)","(5, 14)","(6, 14)","(6, 14)","(6, 14)"
7,,,"(3, 13)","(4, 13)","(4, 14)","(5, 14)","(5, 14)","(5, 14)","(5, 15)","(5, 15)","(6, 15)","(6, 16)","(6, 16)","(6, 16)","(6, 16)","(6, 16)"
8,,,,"(4, 14)","(5, 14)","(5, 15)","(5, 15)","(6, 16)","(6, 16)","(6, 16)","(6, 16)","(6, 17)","(7, 17)","(7, 17)","(7, 17)","(7, 17)"
9,,,,,"(5, 15)","(5, 16)","(6, 16)","(6, 16)","(6, 17)","(7, 17)","(7, 18)","(7, 18)","(7, 18)","(8, 18)","(8, 18)","(8, 18)"
10,,,,,,"(6, 16)","(6, 17)","(7, 17)","(7, 18)","(7, 18)","(7, 18)","(8, 19)","(8, 19)","(8, 19)","(8, 20)","(9, 20)"
11,,,,,,,"(7, 17)","(7, 18)","(7, 19)","(8, 19)","(8, 19)","(8, 20)","(9, 20)","(9, 20)","(9, 21)","(9, 21)"
12,,,,,,,,"(7, 19)","(8, 19)","(8, 20)","(8, 20)","(9, 21)","(9, 21)","(9, 21)","(10, 22)","(10, 22)"
13,,,,,,,,,"(8, 20)","(9, 20)","(9, 21)","(9, 21)","(10, 22)","(10, 22)","(10, 23)","(10, 23)"
14,,,,,,,,,,"(9, 21)","(9, 22)","(10, 22)","(10, 23)","(10, 23)","(11, 23)","(11, 24)"
15,,,,,,,,,,,"(10, 22)","(10, 23)","(11, 23)","(11, 24)","(11, 24)","(12, 25)"
16,,,,,,,,,,,,"(11, 23)","(11, 24)","(11, 25)","(12, 25)","(12, 25)"
17,,,,,,,,,,,,,"(11, 25)","(12, 25)","(12, 26)","(13, 26)"
18,,,,,,,,,,,,,,"(12, 26)","(13, 26)","(13, 27)"
19,,,,,,,,,,,,,,,"(13, 27)","(13, 27)"
20,,,,,,,,,,,,,,,,"(14, 28)"
""")
_runs_test_table = pd.read_csv(_raw_r_table, index_col=0)


def add_margin(ax, x=0.05, y=0.05):

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xmargin = (xlim[1]-xlim[0])*x
    ymargin = (ylim[1]-ylim[0])*y

    ax.set_xlim(xlim[0]-xmargin, xlim[1]+xmargin)
    ax.set_ylim(ylim[0]-ymargin, ylim[1]+ymargin)


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


def c_of_c_test(r, n, beta1=0, alpha=.05, precision=4, show=True, option='two-tail', **kwargs):
    """
    two-tail, right, left
    kwargs needs to contain b1 and std_err if used
    """
    opt = option.lower()[0]
    t_value = r * ((n-2)/(1 - r**2)) ** 0.5
    if beta1 != 0:
        t_value = (kwargs['b1'] - beta1) / kwargs['std_err']
    t_critical = stats.t.ppf(1 - alpha/2, n - 2)
    p_value = stats.t.sf(np.abs(t_value), n - 2)*2
    if opt == 't':
        # two-tail test
        option = 'Two-Tail Test'
        p_value = (1 - stats.t.cdf(t_value, df=n-2)) * 2
        if (p_value > 1):
            p_value = (stats.t.cdf(t_value, df=n-2)) * 2
        t_critical = stats.t.ppf(1 - alpha/2, df=n-2)
    else:
        if opt == 'l':
            option = 'One-Tail Test (left tail)'
            p_value = stats.t.cdf(t_value, df=n - 2)
            t_critical = stats.t.ppf(alpha, df=n - 2)
        elif opt == 'r':
            option = 'One-Tail Test (right tail)'
            p_value = stats.t.sf(t_value, df=n - 2)
            t_critical = stats.t.ppf(1 - alpha, df=n - 2)

    des = f"""r = {r:.{precision}f}
t (critical value) = {t_critical:.{precision}f}
t (observed value)  = {t_value:.{precision}f}
p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})
    """
    if show:
        print(des)
    return t_value, t_critical, p_value, option


def SimpleLinearRegression(Independence=None, Dependence=None, df=None, alpha=0.05, precision=4, show_summary=True, plot=False, slope_option='two-tail', beta1=0, coeff_option='two-tail', kwargs={'x': 0.02, 'y': 0.00, 'title': 'Scatter Plot'}):
    """
    df_result = smf.ols(f'{Dependence}~ {Independence}', data=df).fit()
    """
    slope, intercept, r_value, p_value, std_err_sb1 = stats.linregress(
        df[Independence], df[Dependence])
    flag = p_value < alpha
    result_dict = {'slope': slope, 'intercept': intercept,
                   'r_value': r_value, 'p_value': p_value}
    fig, ax = plt.subplots()

    ax = sns.regplot(x=Independence, y=Dependence, data=df, ci=None, scatter_kws={'color': 'dodgerblue'}, line_kws={
        'color': '#ffaa77', 'label': f"y = {intercept:.4f} + {slope:.4f} x"})

    add_margin(ax, x=kwargs['x'], y=kwargs['y'])  # Call this after tsplot
    result_dict['fig'] = fig
    result_dict['ax'] = ax
    if plot:
        plt.legend()
        plt.title(kwargs['title'])
        if 'xlabel' in kwargs:
            plt.xlabel(kwargs['xlabel'])
        else:
            plt.xlabel(Independence)
        if 'ylabel' in kwargs:
            plt.ylabel(kwargs['ylabel'])
        else:
            plt.ylabel(Dependence)
        plt.show()
    else:
        plt.close(fig)

    df_result = smf.ols(f'{Dependence}~ {Independence}', data=df).fit()
    result_dict['result'] = df_result
    s_e = df_result.mse_resid ** 0.5
    ssr_value = df_result.ssr
    r_square = r_value ** 2
    s_t_value, s_t_critical, s_p_value, s_option = c_of_c_test(
        r_value, df_result.nobs, beta1=beta1, alpha=alpha, precision=precision, show=False, option=slope_option, kwargs={'b1': slope, 'std_err': std_err_sb1})
    s_flag = s_p_value < alpha
    t_t_value, t_t_critical, t_p_value, t_option = c_of_c_test(
        r_value, df_result.nobs, alpha=alpha, precision=precision, show=False, option=coeff_option)
    if t_option == 'Two-Tail Test':
        t_p_value = p_value

    ci_b1 = df_result.conf_int(alpha)[1:].values[0]
    result_dict['ci_b1'] = ci_b1

    if show_summary:
        print(df_result.summary())
        print()
    results = f"""======= Simple Linear Regression Results =======
Dep. Variable: {Dependence}
No. of Observations: {int(df_result.nobs)}
Standard Error: {s_e:.{precision}f}
SSR: {ssr_value:.{precision}f}
R-square: {r_square:.{precision}f}

Estimated model: y = {intercept:.{precision}f} + {slope:.{precision}f} x

**** t-Test of Slope <{s_option}> ****
t (observed value): {s_t_value:.{precision}f}
t (critical value): {s_t_critical:.{precision}f}
b1 (slope): {slope:.{precision}f}
p-value: {s_p_value:.{precision}f} ({inter_p_value(s_p_value)})

Reject H_0 (Has some kind of relationship between two variables) → {s_flag}

{(1-alpha) * 100}% confidence interval = [{ci_b1[0]:.4f}, {ci_b1[1]:.4f}]

**** t-Test of Correlation Coefficient <{t_option}> ****
t (observed value): {t_t_value:.{precision}f}
t (critical value): {t_t_critical:.{precision}f}
r: {r_value:.{precision}f}
p-value: {t_p_value:.{precision}f} ({inter_p_value(t_p_value)})
Reject H_0 (Has Linear Relationship) → {flag}"""
    print(results)
    result_dict['description'] = results

    return result_dict
