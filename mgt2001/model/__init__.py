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
import statsmodels.stats.outliers_influence as sso
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
    kwargs needs to contain b1 and std_err if used: 
    >>> if beta1 != 0:
    >>>    t_value = (kwargs['b1'] - beta1) / kwargs['std_err']
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


def SimpleLinearRegressionPrediction(Independence=None, Dependence=None, df=None, x=None, alpha=0.05, plot=True, kwargs={'title': 'Two Types of Intervals', 'xlabel': 'Independent Variable', 'ylabel': 'Dependent Variable'}):
    """

    """
    x1 = x
    if type(Independence) == str:
        x, y = df[Independence], df[Dependence]
    else:
        x, y = Independence, Dependence
    x_new = np.array([1, x1])
    X2 = sm.add_constant(x)
    olsmod = sm.OLS(y, X2)
    result_reg = olsmod.fit()
    y_head = np.dot(result_reg.params, x_new)
    (t_minus, t_plus) = stats.t.interval(
        alpha=(1.0 - alpha), df=result_reg.df_resid)
    cov_mat1 = np.cov(y, x)
    x_bar = x.mean()
    core1 = (1 / result_reg.nobs +
             (x1 - x_bar) ** 2 / (result_reg.nobs - 1) / cov_mat1[1, 1]) ** 0.5
    core2 = (1 + 1 / result_reg.nobs +
             (x1 - x_bar) ** 2 / (result_reg.nobs - 1) / cov_mat1[1, 1]) ** 0.5
    lower_bound = y_head + t_minus * (result_reg.mse_resid ** 0.5) * core1
    upper_bound = y_head + t_plus * (result_reg.mse_resid ** 0.5) * core1
    half_interval = t_plus * (result_reg.mse_resid ** 0.5) * core1
    lower_bound2 = y_head + t_minus * (result_reg.mse_resid ** 0.5) * core2
    upper_bound2 = y_head + t_plus * (result_reg.mse_resid ** 0.5) * core2
    half_interval2 = t_plus * (result_reg.mse_resid ** 0.5) * core2
    des = f"""======= Making Prediction =======
Make CI and PI predictions at mean of x = {x1}
y_head = {y_head}

Confidence interval for mean: [{lower_bound:.4f}, {upper_bound:.4f}]
    or {y_head:.4f}  ± {half_interval:.4f}
Prediction interval (confidence interval) for Exact Value: [{lower_bound2:.4f}, {upper_bound2:.4f}]
    or {y_head:.4f}  ± {half_interval2:.4f}
        """

    print(des)
    CI_PI = {'CI': [lower_bound, upper_bound],
             'PI': [lower_bound2, upper_bound2]}
    try:
        df_sorted = df.sort_values([Independence])
        df_res = smf.ols(f'{Dependence}~ {Independence}', data=df_sorted).fit()
        x = df_sorted[Independence].values
        y = df_sorted[Dependence].values
        fig, ax = plt.subplots()
        st, data, ss3 = sso.summary_table(df_res, alpha=alpha)
        fittedvalues = data[:, 2]
        predict_mean_se = data[:, 3]
        predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T
        predict_ci_low, predict_ci_upp = data[:, 6:8].T
        plt.plot(x, y, 'o', color='gray')
        plt.plot(x, fittedvalues, 'g-', lw=0.5, label='Model')
        plt.plot(x, predict_mean_ci_low, 'r-',
                 lw=0.4, label='Confidence Interval')
        plt.plot(x, predict_mean_ci_upp, 'r-', lw=0.4)
        plt.plot(x, predict_ci_low, 'b--', lw=0.4,
                 label='Prediction Interval')
        plt.plot(x, predict_ci_upp, 'b--', lw=0.4)
        circle_rad = 12
        ax.plot((x1), (y_head), 'o', color='gold',
                label=f'$(x, \hat y)$ = ({x1}, {y_head:.4f})')
        ax.plot((x1), (y_head), 'o', ms=circle_rad *
                1.2, mec='gold', mfc='none', mew=2)
        plt.title(kwargs['title'])
        plt.xlabel(kwargs['xlabel'])
        plt.ylabel(kwargs['ylabel'])
        plt.legend()
        ax.legend(loc='center left', bbox_to_anchor=(
            1, 0.5), fancybox=True, shadow=True)
        CI_PI['fig'] = fig
        CI_PI['ax'] = ax
        if plot:
            plt.show()
        else:
            plt.close(fig)
    except:
        print('Indepent variable (x) and Dependent variable (y) should be passed in as "strings". Additional DataFrame is also required.')
    return CI_PI


def SimpleLinearRegressionOutlier(Independence=None, Dependence=None, df=None, outlier=True, influential=True):
    df_res = smf.ols(f'{Dependence}~ {Independence}', data=df).fit()
    simple_table, data, ss3 = sso.summary_table(df_res, alpha=0.05)
    std_resid = data[:, 10]
    return_dict = {}
    if outlier:
        df_w_std_resid = df.copy().reset_index().rename(
            columns={'index': 'ID'})
        df_w_std_resid['ID'] += 1
        df_w_std_resid['Std. Resid'] = std_resid
        filter = (df_w_std_resid['Std. Resid'] < -
                  2) | (df_w_std_resid['Std. Resid'] > 2)
        df_w_std_resid = df_w_std_resid[filter]
        # display(df_w_std_resid)
        return_dict['df_w_std_resid'] = df_w_std_resid

        y_pre = data[:, 2]  # x

        fig, ax = plt.subplots()

        plt.plot(y_pre, std_resid, 'o', color='gray')
        plt.axhline(y=0, color='blue')
        plt.axhline(y=2, color='red')
        plt.axhline(y=-2, color='red')

        circle_rad = 12
        for i in df_w_std_resid.index:
            x_id = i
            y_id = data[x_id, 2]
            sr_id = data[x_id, 10]

            ax.plot((y_id), (sr_id), 'o',
                    ms=circle_rad * 1.5, mec='violet', mfc='none', mew=2)

        plt.title('Standardized Residual Plot - Outliers in Violet Circle')
        plt.xlabel('Predicted Dependent Variable')
        plt.ylabel('Standardized Residual')

        return_dict['outlier-fig'] = fig
        return_dict['outlier-ax'] = ax

        plt.show()

    if influencial:
        df_w_h = df.copy().reset_index().rename(columns={'index': 'ID'})
        x_data = df[Independence].values
        y_data = df[Dependence].values
        cov_mat1 = np.cov(y_data, x_data)
        x_data_bar = x_data.mean()
        nobs = len(x_data)
        h_val = 1 / nobs + (x_data - x_data_bar) ** 2 / \
            (nobs - 1) / cov_mat1[1, 1]
        df_w_h['h (leverage)'] = h_val
        filter = (df_w_h['h (leverage)'] > 6 / nobs)
        df_w_h = df_w_h[filter]
        return_dict['df_w_h'] = df_w_h

        y_pre = data[:, 2]  # x

        fig, ax = plt.subplots()

        plt.plot(y_pre, std_resid, 'o', color='gray')
        plt.axhline(y=0, color='blue')
        plt.axhline(y=2, color='red')
        plt.axhline(y=-2, color='red')

        # cir = plt.Circle((y_id, sr_id), 0.2, color='y',fill=False)
        circle_rad = 12

        for i in df_w_h.index:
            x_id = i
            y_id = data[x_id, 2]
            sr_id = data[x_id, 10]
        #     x_id, y_id, sr_id

            ax.plot((y_id), (sr_id), 'o',
                    ms=circle_rad * 1.5, mec='springgreen', mfc='none', mew=2)
        # ax.set_aspect('equal', adjustable='datalim')
        # ax.add_patch(cir)

        plt.title(
            'Standardized Residual Plot - Influential Observations in Spring Green')
        plt.xlabel(
            'Predicted Dependent Variable')
        plt.ylabel('Standardized Residual')
        # plt.legend()
        return_dict['inf-fig'] = fig
        return_dict['inf-ax'] = ax
        plt.show()

    # Scatter Plot with Two Variables and Fixed Margins (Seaborn included)
    fig, ax = plt.subplots()

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df[Independence], df[Dependence])  # order matters

    ax = sns.regplot(x=Independence, y=Dependence, data=df, ci=None, scatter_kws={'color': 'dodgerblue'}, line_kws={
                     'color': '#ffaa77', 'label': f"y = {intercept:.4f} + {slope:.4f} x"})

    circle_rad = 12

    try:
        for i in df_w_std_resid.index:
            x_id = i
            y_id = data[x_id, 1]
            x_value = df[Independence][x_id]

            ax.plot((x_value), (y_id), 'o',
                    ms=circle_rad * 1.5, mec='violet', mfc='none', mew=2)
    except:
        pass
    try:
        for i in df_w_h.index:
            x_id = i
            y_id = data[x_id, 1]
            x_value = df[Independence][x_id]

            ax.plot((x_value), (y_id), 'o',
                    ms=circle_rad * 1.5, mec='springgreen', mfc='none', mew=2)
    except:
        pass

    plt.legend()

    plt.title('Scatter Plot')
    plt.xlabel(Independence)
    plt.ylabel(Dependence)

    add_margin(ax, x=0.02, y=0.00)  # Call this after tsplot

    return_dict['all-fig'] = fig
    return_dict['all-ax'] = ax
    plt.show()

    return return_dict


def SimpleLinearRegression(Independence=None, Dependence=None, df=None, alpha=0.05, precision=4, show_summary=True, plot=False, test=True, slope_option='two-tail', beta1=0, coeff_option='two-tail', kwargs={'x': 0.02, 'y': 0.00, 'title': 'Scatter Plot'}):
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

Estimated model: y = {intercept:.{precision}f} + {slope:.{precision}f} x"""

    if test == True:
        results += f"""
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


def runs_test(x, cutoff='median', alpha=0.05, precision=4):
    if cutoff == 'median':
        cutoff = np.median(x)
    elif cutoff == 'mean':
        cutoff = np.mean(x)

    indicator = (x >= cutoff)
    n1 = np.sum(x >= cutoff)
    n2 = np.sum(x < cutoff)
    n_runs = 1

    for i in range(1, len(indicator)):
        if indicator[i] != indicator[i - 1]:
            n_runs += 1

    if n1 <= 20 and n2 <= 20:
        display(_runs_test_table)
        if n1 > n2:
            big_n = n1
            small_n = n2
        else:
            big_n = n2
            small_n = n1

        res = (_runs_test_table.loc[small_n, str(big_n)])
        lb, ub = tuple(map(int, res.strip('()').split(', ')))
        if n_runs < ub and n_runs > lb:
            flag = False
        else:
            flag = True
        des = f"""======= Runs Test =======
(Both n1 ({n1}) and n2 ({n2}) <= 20)
Runs = {n_runs}
Lower bound and Upper bound = [{lb}, {ub}]
Reject H_0 (Randomness does not exist) → {flag}
"""
        print(des)
        return flag, ub, lb

    else:
        mu_r = 2 * n1 * n2 / (n1 + n2) + 1
        sigma_r = math.sqrt(2 * n1 * n2 * (2 * n1 * n2 -
                                           n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1)))
        z_value = (n_runs - mu_r) / sigma_r
        p_value = (1 - stats.norm.cdf(z_value)) * 2
        if (p_value > 1):
            p_value = (stats.norm.cdf(z_value)) * 2
        flag = p_value < alpha
        des = f"""======= Runs Test =======
(n1 ({n1}) or n2 ({n2}) > 20)
Runs = {n_runs}
runs_exp (mu_r) = {mu_r:.4f}
std (sigma_r) = {sigma_r:4f}

z-value (observed statistic) = {z_value:.4f}
p-value = {p_value:.4f}
Reject H_0 (Randomness does not exist) → {flag}
"""
        print(des)

        return z_value, p_value
