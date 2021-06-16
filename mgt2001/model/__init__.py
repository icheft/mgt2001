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
from statsmodels.stats.stattools import durbin_watson as sdw
from . import table
from .model_building import *
from .model_building import _color_palette


_runs_test_table = pd.read_csv(table._raw_r_table, index_col=0)

_dwcrit_table = pd.read_csv(table._dwcrit_table, index_col=0)
_dwcrit_table25 = pd.read_csv(table._dwcrit_table25, index_col=0)


def durbin_watson(std_resid, n=None, k=None, option='positive', precision=4):
    """
    only detect first order autocorrelation between conseutive residuals in a time series

    + option = 'positive' / 'right-tail'
    + option = 'two-tail
    + option = 'negative' / 'left-tail'
    """
    rej_h_0_prompt = ""
    dwcrit_table = _dwcrit_table
    alpha = 0.05
    if option.lower()[0] == 'p' or option.lower()[0] == 'r':
        opt = 'r'
        rej_h_0_prompt = 'The data are positively first-order correlated'
    elif option.lower()[0] == 'n' or option.lower()[0] == 'l':
        opt = 'l'
        rej_h_0_prompt = 'The data are negatively first-order correlated'
    else:
        opt = 't'
        alpha = 0.025
        dwcrit_table = _dwcrit_table25
        rej_h_0_prompt = 'The data are first-order correlated'

    d = sdw(std_resid)
    d_l, d_u = dwcrit_table[(dwcrit_table['Sample Size'] == n) & (dwcrit_table['Number of terms (including the intercept)'] == k + 1)][['D  L',
                                                                                                                                        'D  U']].values.tolist()[0]

    d_result = f'''
d = {d:.{precision}f}
(d_l, d_u) = ({d_l:.{precision}f}, {d_u:.{precision}f})

with n = {n} and k = {k}
'''

    if opt == 'r':
        if d < d_l and d >= 0:
            flag = True
        elif d > d_u and d <= 4:
            flag = False
        elif d >= d_l and d <= d_u:
            flag = 'Inconclusive'
            rej_h_0_prompt = 'No meaning'
        else:
            print('Error occurred')
            print(d_result)
            return d, d_l, d_u

    elif opt == 'l':
        if d > 4 - d_l and d <= 4:
            flag = True
        elif d < 4 - d_u and d >= 2:
            flag = False
        elif d >= 4 - d_u and d <= 4 - d_l:
            flag = 'Inconclusive'
            rej_h_0_prompt = 'No meaning'
        else:
            print('Error occurred')
            print(d_result)
            return d, d_l, d_u
    elif opt == 't':
        if d < d_l or d > 4 - d_l:
            flag = True
        elif (d >= 4 - d_u and d <= 4 - d_l) or (d <= d_u and d >= d_l):
            flag = 'Inconclusive'
            rej_h_0_prompt = 'No meaning'
        elif d > d_u and d < 4 - d_u:
            flag = False
        else:
            print('Error occurred')
            print(d_result)
            return d, d_l, d_u

    results = f"""======== Durbin-Watson Test for First Order Correlation =======
d = {d:.{precision}f}
(d_l, d_u) = ({d_l:.{2}f}, {d_u:.{2}f})

with n = {n} and k = {k} / alpha = {alpha}

Reject H_0 ({rej_h_0_prompt}) → {flag}
"""
    print(results)
    return d, d_l, d_u


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


def multi_scatter_plot(row, col, df, x_names, y_name, figsize=(13, 7), hspace=0.2, wspace=0.2, scatter_show=True, corr_show=True):
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    y_var = df[y_name]

    for i, name in enumerate(x_names):
        ax = fig.add_subplot(row, col, i + 1)
        # data = df[value_name][df[treatment_name] == name]
        x_var = df[name].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x_var, y_var)  # order matters

        ax = sns.regplot(x=name, y=y_name, data=df, ci=None, scatter_kws={'color': 'dodgerblue'}, line_kws={
                         'color': '#ffaa77', 'label': f"$\hat y = {intercept:.4f} + {slope:.4f} x$"})

        plt.legend()

        plt.title(f'Scatter Plot for {name} vs {y_name}')
        plt.xlabel(name)
        plt.ylabel(y_name)

        add_margin(ax, x=0.02, y=0.00)  # Call this after tsplot

    fig.tight_layout()
    if scatter_show:
        plt.show()
    else:
        plt.close()

    df_corr = df[[y_name] + x_names]
    corr = df_corr.corr()
    fig, ax = plt.subplots()
    _ = sns.heatmap(corr, annot=True)
    if corr_show:
        plt.show()
    else:
        plt.close()


def multicollinearity(df, x_names, y_name):
    df_corr = df[[y_name] + x_names]
    corr = df_corr.corr()
    coef = corr.iloc[:, 0].values

    def _highlight(val):
        color = 'salmon' if (abs(val) >= 0.7 and val !=
                             1 and val not in coef) else 'default'
        return 'background-color: %s' % color

    style_df = corr.style.applymap(_highlight)
    return style_df


def SimpleLinearRegressionPrediction(x_name=None, y_name=None, df=None, x=None, alpha=0.05, plot=True, kwargs={'title': 'Two Types of Intervals', 'xlabel': 'Independent Variable', 'ylabel': 'Dependent Variable'}):
    """

    """
    x1 = x
    if type(x_name) == str:
        x, y = df[x_name], df[y_name]
    else:
        x, y = x_name, y_name
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
        df_sorted = df.sort_values([x_name])
        df_res = smf.ols(f'{y_name}~ {x_name}', data=df_sorted).fit()
        x = df_sorted[x_name].values
        y = df_sorted[y_name].values
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


def SimpleLinearRegressionOutlier(x_name=None, y_name=None, df=None, outlier=True, influential=True, plot=True, display_df=True, **kwargs):
    """
    kwargs:

    + Pass in xlabel to specify the x label for individual residual plot
    + pass in s_xlabel to specify the x label for the scatter plot
    + pass in s_ylabel to specify the y label for the scatter plot
    """

    df_res = smf.ols(f'{y_name}~ {x_name}', data=df).fit()
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
        return_dict['outlier_filter'] = filter

        if display_df:
            print('Outliers:')
            display(df_w_std_resid)

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
        ax.plot([], [], 'o', ms=circle_rad * 0.5, mec='violet',
                mfc='none', mew=2, label='Outliers')

        plt.title('Standardized Residual Plot - Outliers in Violet Circle')
        try:
            xlabel = kwargs['xlabel']
        except KeyError:
            xlabel = f'Predicted {y_name}'
        plt.xlabel(xlabel)
        plt.ylabel('Standardized Residual')

        return_dict['outlier-fig'] = fig
        return_dict['outlier-ax'] = ax

        if plot:
            plt.show()
        else:
            plt.close()

    if influential:
        df_w_h = df.copy().reset_index().rename(columns={'index': 'ID'})
        df_w_h['ID'] += 1
        x_data = df[x_name].values
        y_data = df[y_name].values
        cov_mat1 = np.cov(y_data, x_data)
        x_data_bar = x_data.mean()
        nobs = len(x_data)
        h_val = 1 / nobs + (x_data - x_data_bar) ** 2 / \
            (nobs - 1) / cov_mat1[1, 1]
        df_w_h['h (leverage)'] = h_val
        filter = (df_w_h['h (leverage)'] > 6 / nobs)
        df_w_h = df_w_h[filter]
        return_dict['df_w_h'] = df_w_h
        return_dict['h cv'] = 6 / nobs
        return_dict['inf_filter'] = filter

        if display_df:
            print('Influential Observations:')
            display(df_w_h)

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
        ax.plot([], [], 'o',
                ms=circle_rad * 0.5, mec='springgreen', mfc='none', mew=2, label='Influential Observations')

        plt.title(
            f'Standardized Residual Plot - Influential Observations ($h_i > {6 / nobs}$) in Spring Green')
        try:
            xlabel = kwargs['xlabel']
        except KeyError:
            xlabel = f'Predicted {y_name}'
        plt.xlabel(xlabel)
        plt.ylabel('Standardized Residual')
        # plt.legend()
        return_dict['inf-fig'] = fig
        return_dict['inf-ax'] = ax
        plt.show()

        if plot:
            plt.show()
        else:
            plt.close()

    # Scatter Plot with Two Variables and Fixed Margins (Seaborn included)
    fig, ax = plt.subplots()

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df[x_name], df[y_name])  # order matters

    ax = sns.regplot(x=x_name, y=y_name, data=df, ci=None, scatter_kws={'color': 'dodgerblue'}, line_kws={
                     'color': '#ffaa77', 'label': f"y = {intercept:.4f} + {slope:.4f} x"})

    circle_rad = 12

    try:
        for i in df_w_std_resid.index:
            x_id = i
            y_id = data[x_id, 1]
            x_value = df[x_name][x_id]

            ax.plot((x_value), (y_id), 'o',
                    ms=circle_rad * 1.5, mec='violet', mfc='none', mew=2)
        ax.plot([], [], 'o', ms=circle_rad * 0.5, mec='violet',
                mfc='none', mew=2, label='Outliers')
    except:
        pass
    try:
        for i in df_w_h.index:
            x_id = i
            y_id = data[x_id, 1]
            x_value = df[x_name][x_id]

            ax.plot((x_value), (y_id), 'o',
                    ms=circle_rad * 1.5, mec='springgreen', mfc='none', mew=2)
        ax.plot([], [], 'o',
                ms=circle_rad * 0.5, mec='springgreen', mfc='none', mew=2, label='Influential Observations')

    except:
        pass

    plt.legend()

    ax.legend(loc='center left', bbox_to_anchor=(
        1, 0.5), fancybox=True, shadow=True)

    plt.title('Scatter Plot')
    try:
        xlabel = kwargs['s_xlabel']
    except KeyError:
        xlabel = f'{x_name}'
    try:
        ylabel = kwargs['s_ylabel']
    except KeyError:
        ylabel = f'{y_name}'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    add_margin(ax, x=0.02, y=0.00)  # Call this after tsplot

    return_dict['all-fig'] = fig
    return_dict['all-ax'] = ax

    if plot:
        plt.show()
    else:
        plt.close()

    return return_dict


def SimpleLinearRegression(x_name=None, y_name=None, df=None, alpha=0.05, precision=4, show_summary=True, plot=False, assessment=True, slope_option='two-tail', beta1=0, coeff_option='two-tail', kwargs={'x': 0.02, 'y': 0.00, 'title': 'Scatter Plot'}):
    """
    df_result = smf.ols(f'{y_name}~ {x_name}', data=df).fit()
    """
    slope, intercept, r_value, p_value, std_err_sb1 = stats.linregress(
        df[x_name], df[y_name])
    # flag = p_value < alpha
    result_dict = {'slope': slope, 'intercept': intercept,
                   'r_value': r_value, 'p_value': p_value}
    fig, ax = plt.subplots()

    ax = sns.regplot(x=x_name, y=y_name, data=df, ci=None, scatter_kws={'color': 'dodgerblue'}, line_kws={
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
            plt.xlabel(x_name)
        if 'ylabel' in kwargs:
            plt.ylabel(kwargs['ylabel'])
        else:
            plt.ylabel(y_name)
        plt.show()
    else:
        plt.close(fig)

    df_result = smf.ols(f'{y_name}~ {x_name}', data=df).fit()
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
    flag = t_p_value < alpha
    ci_b1 = df_result.conf_int(alpha)[1:].values[0]
    result_dict['ci_b1'] = ci_b1

    if show_summary:
        print(df_result.summary())
        print()
    results = f"""======= Simple Linear Regression Results =======
Dep. Variable: {y_name}
No. of Observations: {int(df_result.nobs)}
Mean of Dep. Variable: {np.mean(df[y_name]):.{precision}f}
Standard Error: {s_e:.{precision}f}
SSR: {ssr_value:.{precision}f}
R-square: {r_square:.{precision}f}

Estimated model: y = {intercept:.{precision}f} + {slope:.{precision}f} x"""

    if assessment == True:
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

    return result_dict, results


def MultipleRegressionPrediction(x_names=None, y_name=None, df=None, x1=None, alpha=0.05, precision=4, ts=False, **kwargs):
    """
    if ts = True: x1 is required to be a list (iterable)
    """
    # print("make CI and PI prediction at mean of x = ", x1)
    if ts:
        tdf = df.copy()
        df_result = kwargs['df_result']
        time_name = kwargs['time_name']
        _, data, _ = sso.summary_table(df_result, alpha=0.05)
        new_t = np.array(x1)
        if len(x1) == 0:
            tdf[f'Pre_{y_name}'] = data[:, 2]
            return tdf
        elif len(x1) == 1:
            search_table = pd.DataFrame(x1, columns=x_names)
            x1 = [1] + x1[0]
            new_t = np.array([x1])
            # print(new_t)
        else:
            search_table = pd.DataFrame(x1, columns=x_names)
            new_t = sm.add_constant(new_t)

        pred_y = df_result.predict(new_t)
        tdf[f'Pre_{y_name}'] = data[:, 2]
        # tdf[x_name] = np.append(tdf[x_name], new_t)
        for i in range(search_table.shape[0]):
            append_dict = {}
            for x in x_names:
                print(x)
                append_dict[x] = search_table.iloc[i, :][x]
            append_dict[f'Pre_{y_name}'] = pred_y[i]
            tdf = tdf.append(
                append_dict, ignore_index=True)

        return tdf

    else:
        x1.insert(0, 1)
        X_data_T = np.array(df[x_names])
        X_data2 = sm.add_constant(X_data_T)
        olsmod = sm.OLS(df[y_name], X_data2)
        result_reg = olsmod.fit()
        # print(result_reg.params, x1)
        y_head = np.dot(result_reg.params, x1)
        # print("y_head = ", y_head)
        (t_minus, t_plus) = stats.t.interval(
            alpha=(1.0 - alpha), df=result_reg.df_resid)
        core1 = (result_reg.mse_resid * np.matmul(x1,
                                                  np.linalg.solve(np.matmul(X_data2.T, X_data2), x1))) ** 0.5
        lower_bound = y_head + t_minus * core1
        upper_bound = y_head + t_plus * core1
        # print("confidence interval of mean = [%0.4f, %0.4f] " % (
        #     lower_bound, upper_bound))
        core2 = (result_reg.mse_resid * (1 + np.matmul(x1,
                                                       np.linalg.solve(np.matmul(X_data2.T, X_data2), x1)))) ** 0.5
        lower_bound2 = y_head + t_minus * core2
        upper_bound2 = y_head + t_plus * core2

        result = f"""======= Making Prediction =======
    Make CI and PI predictions at mean of x = {x1}
    y_head = {y_head}

    ----------------
    {pd.DataFrame(result_reg.params, columns=['coef'])}
    ----------------

    Confidence interval for mean: [{lower_bound:.{precision}f}, {upper_bound:.{precision}f}]
    Prediction interval (confidence interval) for Exact Value: [{lower_bound2:.{precision}f}, {upper_bound2:.{precision}f}]
    """
        print(result)
        CI_PI = {'CI': [lower_bound, upper_bound],
                 'PI': [lower_bound2, upper_bound2]}
        return CI_PI


def MultipleRegressionOutlier(x_names=None, y_name=None, df=None, std_resid=None, outlier=True, influential=True, cook=True, alpha=.05, plot=True, display_df=True, **kwargs):
    """
    Pass in x_names, y_name, and df as well

    kwargs:

    + Pass in xlabel to specify the x label for individual residual plot
    + pass in s_xlabel to specify the x label for the scatter plot
    + pass in s_ylabel to specify the y label for the scatter plot
    """

    # df_res = smf.ols(f'{y_name}~ {x_name}', data=df).fit()
    y_data = df[y_name]
    X_data_T = np.array(df[x_names])
    X_data = pd.DataFrame(X_data_T, columns=x_names)
    X_data_update = sm.add_constant(X_data)
    olsmod = sm.OLS(y_data, X_data_update)
    reg_result = olsmod.fit()
    if std_resid is not None:
        pass
    else:
        reg_result = olsmod.fit()
        simple_table, data, ss3 = sso.summary_table(reg_result, alpha=alpha)
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
        return_dict['outlier_filter'] = filter

        if display_df:
            print('Outliers:')
            display(df_w_std_resid)

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
        ax.plot([], [], 'o', ms=circle_rad * 0.5, mec='violet',
                mfc='none', mew=2, label='Outliers')

        plt.title('Standardized Residual Plot - Outliers in Violet Circle')
        try:
            xlabel = kwargs['xlabel']
        except KeyError:
            xlabel = f'Predicted {y_name}'
        plt.xlabel(xlabel)
        plt.ylabel('Standardized Residual')

        return_dict['outlier-fig'] = fig
        return_dict['outlier-ax'] = ax

        if plot:
            plt.show()
        else:
            plt.close()

    if influential:
        x_data = df[x_names].values
        y_data = df[y_name].values
        x_data = sm.add_constant(x_data)
        H = np.matmul(x_data, np.linalg.solve(
            np.matmul(x_data.T, x_data), x_data.T))
        df_w_h = df.copy().reset_index().rename(columns={'index': 'ID'})
        df_w_h['ID'] += 1
        df_w_h['h_ii'] = np.diagonal(H)
        k = len(x_names)
        n = len(df_w_h['h_ii'])
        h_level = 3 * (k+1) / n
        # print("h_level = ", h_level)
        # print(" \n")
        filter = (df_w_h['h_ii'] > h_level)
        # print("Influential Observations by hi = \n")
        # print(df_w_h['h_ii'].loc[filter])
        df_w_h = df_w_h[filter]
        return_dict['df_w_h'] = df_w_h
        return_dict['h cv'] = h_level
        return_dict['inf_filter'] = filter

        if display_df:
            print('Influential Observations:')
            display(df_w_h)

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
        ax.plot([], [], 'o',
                ms=circle_rad * 0.5, mec='springgreen', mfc='none', mew=2, label='Influential Observations')
        ii = "ii"
        plt.title(
            f'Standardized Residual Plot - Influential Observations ($h_{ii} > {h_level}$) in Spring Green')
        try:
            xlabel = kwargs['xlabel']
        except KeyError:
            xlabel = f'Predicted {y_name}'
        plt.xlabel(xlabel)
        plt.ylabel('Standardized Residual')
        # plt.legend()
        return_dict['inf-fig'] = fig
        return_dict['inf-ax'] = ax
        plt.show()

        if plot:
            plt.show()
        else:
            plt.close()

    if cook:
        s2_e = reg_result.mse_resid
        k = reg_result.df_model
        y_a = data[:, 1]
        y_f = data[:, 2]
        x_data = df[x_names].values
        y_data = df[y_name].values
        x_data = sm.add_constant(x_data)
        H = np.matmul(x_data, np.linalg.solve(
            np.matmul(x_data.T, x_data), x_data.T))
        df_w_h = df.copy().reset_index().rename(columns={'index': 'ID'})
        df_w_h['ID'] += 1
        df_w_h['h_ii'] = np.diagonal(H)
        h_i = df_w_h['h_ii']

        CD_arr = np.square(y_a - y_f) / s2_e / (k - 1) * \
            h_i / np.square(1 - h_i)

        CD = np.array(CD_arr)
        df_w_cd = df.copy().reset_index().rename(columns={'index': 'ID'})
        df_w_cd['ID'] += 1
        df_w_cd['CD'] = CD
        # display(df_w_cd)
        filter = (df_w_cd['CD'] > 1)
        # print("Influential Observations by Cook's Distances = \n")
        df_w_cd = df_w_cd[filter]
        return_dict['df_w_cd'] = df_w_cd
        # return_dict['h cv'] = h_level
        return_dict['cd_filter'] = filter

        if display_df:
            print('Influential Observations by Cook\'s Distance:')
            display(df_w_cd)

        y_pre = data[:, 2]  # x

        fig, ax = plt.subplots()

        plt.plot(y_pre, std_resid, 'o', color='gray')
        plt.axhline(y=0, color='blue')
        plt.axhline(y=2, color='red')
        plt.axhline(y=-2, color='red')

        # cir = plt.Circle((y_id, sr_id), 0.2, color='y',fill=False)
        circle_rad = 12

        for i in df_w_cd.index:
            x_id = i
            y_id = data[x_id, 2]
            sr_id = data[x_id, 10]
        #     x_id, y_id, sr_id

            ax.plot((y_id), (sr_id), 'o',
                    ms=circle_rad * 1.5, mec='goldenrod', mfc='none', mew=2)
        # ax.set_aspect('equal', adjustable='datalim')
        # ax.add_patch(cir)
        ax.plot([], [], 'o',
                ms=circle_rad * 0.5, mec='goldenrod', mfc='none', mew=2, label='Influential Observations by Cook\'s Distance')

        plt.title(
            f'Standardized Residual Plot - \nInfluential Observations by Cook\'s Distance ($D_i > 1$) in Golden Rod')
        try:
            xlabel = kwargs['xlabel']
        except KeyError:
            xlabel = f'Predicted {y_name}'
        plt.xlabel(xlabel)
        plt.ylabel('Standardized Residual')
        # plt.legend()
        return_dict['cd-fig'] = fig
        return_dict['cd-ax'] = ax
        plt.show()

        if plot:
            plt.show()
        else:
            plt.close()

    # # Scatter Plot with Two Variables and Fixed Margins (Seaborn included)
    # fig, ax = plt.subplots()

    # # slope, intercept, r_value, p_value, std_err = stats.linregress(
    # #     df[x_name], df[y_name])  # order matters

    # # ax = sns.scatterplot(x=x_name, y=y_name, data=df, ci=None, scatter_kws={'color': 'dodgerblue'}, line_kws={
    # #                  'color': '#ffaa77', 'label': f"y = {intercept:.4f} + {slope:.4f} x"})

    # circle_rad = 12

    # try:
    #     for i in df_w_std_resid.index:
    #         x_id = i
    #         y_id = data[x_id, 1]
    #         x_value = df[x_name][x_id]

    #         ax.plot((x_value), (y_id), 'o',
    #                 ms=circle_rad * 1.5, mec='violet', mfc='none', mew=2)
    #     ax.plot([], [], 'o', ms=circle_rad * 0.5, mec='violet',
    #             mfc='none', mew=2, label='Outliers')
    # except:
    #     pass
    # try:
    #     for i in df_w_h.index:
    #         x_id = i
    #         y_id = data[x_id, 1]
    #         x_value = df[x_name][x_id]

    #         ax.plot((x_value), (y_id), 'o',
    #                 ms=circle_rad * 1.5, mec='springgreen', mfc='none', mew=2)
    #     ax.plot([], [], 'o',
    #             ms=circle_rad * 0.5, mec='springgreen', mfc='none', mew=2, label='Influential Observations')

    # except:
    #     pass

    # plt.legend()

    # ax.legend(loc='center left', bbox_to_anchor=(
    #     1, 0.5), fancybox=True, shadow=True)

    # plt.title('Scatter Plot')
    # try:
    #     xlabel = kwargs['s_xlabel']
    # except KeyError:
    #     xlabel = f'{x_name}'
    # try:
    #     ylabel = kwargs['s_ylabel']
    # except KeyError:
    #     ylabel = f'{y_name}'
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)

    # add_margin(ax, x=0.02, y=0.00)  # Call this after tsplot

    # return_dict['all-fig'] = fig
    # return_dict['all-ax'] = ax

    # if plot:
    #     plt.show()
    # else:
    #     plt.close()

    return return_dict


def MultipleRegressionResidual(x_names=None, y_name=None, df=None, reg_result=None, durbin_watson_test=True, durbin_watson_test_option='two-tail', alpha=0.05, precision=4):
    """
    np.mean(std_resid), np.std(std_resid, ddof=1)
    """
    if reg_result is not None:
        st, data, ss2 = sso.summary_table(reg_result, alpha=alpha)
    else:
        y_data = df[y_name]
        X_data_T = np.array(df[x_names])
        X_data = pd.DataFrame(X_data_T, columns=x_names)
        X_data_update = sm.add_constant(X_data)
        olsmod = sm.OLS(y_data, X_data_update)
        reg_result = olsmod.fit()
    st, data, ss2 = sso.summary_table(reg_result, alpha=alpha)
    y_pre = data[:, 2]
    y_id = data[:, 0]
    std_resid = data[:, 10]
    # std_resid
    stat, p_value = stats.shapiro(std_resid)
    # print('Statistics=%.4f, p=%.4f' % (stat, p))
    result = f'''======= Multiple Regression: Residual Analysis =======
☑ Normality Check: Using Shapiro-Wilk Test

Statistics = {stat:.{precision}f}, p-value = {p_value:.{precision}f} ({inter_p_value(p_value)})
Reject H_0 (Not normally distributed) → {p_value < alpha}

☑ Heteroscedasticity and Homoscedasticity
'''
    print(result)

    fig = plt.figure(figsize=(9, 4))
    fig.subplots_adjust(hspace=0.02, wspace=0.02)

    ax = fig.add_subplot(1, 2, 1)
    # data = df[value_name][df[treatment_name] == name]
    plt.plot(y_pre, std_resid, 'o', color='gray')
    plt.axhline(y=2, color='red', lw=0.8)
    plt.axhline(y=0, color='blue')
    plt.axhline(y=-2, color='red', lw=0.8)
    plt.title('Standardized Residual Plot')
    plt.xlabel('Predicted y value')
    plt.ylabel('Standardized Residual')

    ax = fig.add_subplot(1, 2, 2)
    plt.hist(std_resid, bins='auto')
    plt.title('Histogram of Standardized Residual')
    plt.xlabel('Standardized Residual')
    plt.ylabel('Frequency')
    add_margin(ax, x=0.02, y=0.00)  # Call this after tsplot

    fig.tight_layout()
    plt.show()

    print('\n☑ Non-x_name of the Error Variable Check\n')

    fig, ax = plt.subplots()
    plt.plot(y_id, std_resid, 'o', color='gray')
    plt.axhline(y=2, color='red', lw=0.8)
    plt.axhline(y=0, color='blue')
    plt.axhline(y=-2, color='red', lw=0.8)
    plt.title('Standardized Residual Plot')
    plt.xlabel('Observation No.')
    plt.ylabel('Standardized Residual')
    plt.show()

    runs_test(std_resid, cutoff='median', alpha=alpha, precision=precision)

    if durbin_watson_test:
        durbin_watson(std_resid, n=len(
            std_resid), k=reg_result.df_model, option=durbin_watson_test_option)

    return np.mean(std_resid), np.std(std_resid, ddof=1)


def MultipleRegression(x_names=None, y_name=None, df=None, alpha=0.05, precision=4, show_summary=True, assessment=True, t_test_c=0, t_test_option='two-tail'):
    """
    assessment is equivalent to t_test: boolean
    """
    res_dict = dict()
    y_data = df[y_name]
    X_data_T = np.array(df[x_names])
    X_data = pd.DataFrame(X_data_T, columns=x_names)
    X_data_update = sm.add_constant(X_data)
    olsmod = sm.OLS(y_data, X_data_update)
    df_result = olsmod.fit()
    ssr_value = df_result.ssr
    s_e = df_result.mse_resid ** 0.5
    res_dict['df_result'] = df_result

    st, data, ss2 = sso.summary_table(df_result, alpha=alpha)
    y_pre = data[:, 2]
    y_id = data[:, 0]
    std_resid = data[:, 10]

    res_dict['y_pre'] = y_pre
    res_dict['y_id'] = y_id
    res_dict['std_resid'] = std_resid
    if show_summary:
        print(df_result.summary())
        print()

    # get params = df_result.params
    # np.array(param)
    # coef_table = pd.DataFrame(df_result.summary().tables[1], dtype=str)
    # coef_table.columns = coef_table.iloc[0]
    # coef_table = coef_table.drop(index=0).reset_index(drop=True)
    # coef_col = coef_table['coef']
    coef_col = df_result.params

    # estimated model
    estimated_mod_str = f'ŷ = {coef_col[0]:.{precision}f}'

    for i, c in enumerate(coef_col[1:]):
        estimated_mod_str += f' + {c:.{precision}f} x{i + 1}'

    results = f"""======= Multiple Regression Results =======
Dep. Variable: {y_name}
No. of Observations (n): {int(df_result.nobs)}
No. of Ind. Vairable (k): {int(df_result.df_model)}
Mean of Dep. Variable: {np.mean(df[y_name]):.{precision}f}
Standard Deviation of Dep. Variable: {np.std(df[y_name], ddof=1):.{precision}f}
Standard Error: {s_e:.{precision}f} (ȳ = {np.mean(y_data):.{precision}f})
SSR: {ssr_value:.{precision}f}

R-square: {df_result.rsquared:.{precision}f}
Adjusted R-square: {df_result.rsquared_adj:.{precision}f}
Difference (≤ 0.06 {abs(df_result.rsquared - df_result.rsquared_adj) <= 0.06}): {df_result.rsquared - df_result.rsquared_adj}

Estimated model: {estimated_mod_str}"""
    results += f'''

<F-test>
F(observed value): {df_result.fvalue: .{precision}f}
p-value: {df_result.f_pvalue: .{precision}f} ({inter_p_value(df_result.f_pvalue)})
Reject H_0 (The model is valid: at least one beta_i ≠ 0) → {df_result.f_pvalue < alpha}
'''
    if assessment:
        # alpha =.05
        # t_test_c = 0
        try:
            i = iter(t_test_c)
        except:
            t_test_c = [t_test_c] * (int(df_result.df_model) + 1)
        conf_int_df = df_result.conf_int(alpha=alpha)
        conf_int_df.columns = [f'[{alpha/2}', f'{1 - alpha/2}]']
        t_test_df = pd.DataFrame(df_result.params, columns=['coef (βi)'])
        t_test_df['p-value'] = df_result.pvalues
        t_test_df = t_test_df.merge(
            conf_int_df, left_index=True, right_index=True)
        t_test_df['H_0'] = [
            f'β{i} = {t_test_c[i]}' for i in range(int(df_result.df_model) + 1)]
        (t_test_c > conf_int_df.iloc[:, 0])
        # option = 'two-tail'
        if t_test_option[0].lower() == 't':
            t_test_df['Reject H_0'] = (t_test_c < conf_int_df.iloc[:, 0]) | (
                t_test_c > conf_int_df.iloc[:, 1])
        elif t_test_option[0].lower() == 'l':
            t_test_df['Reject H_0'] = (t_test_c < conf_int_df.iloc[:, 0]) & (
                t_test_c < conf_int_df.iloc[:, 1])
        elif t_test_option[0].lower() == 'r':
            t_test_df['Reject H_0'] = (t_test_c > conf_int_df.iloc[:, 0]) & (
                t_test_c > conf_int_df.iloc[:, 1])
        # t_test_df

        res_dict['t_test_df'] = t_test_df

        results += f'''
<t-test for each βi>
{t_test_df.round(precision)}
'''
    print(results)
    return res_dict, results


def runs_test(x, cutoff='median', alpha=0.05, precision=4, show_table=False):
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
        if show_table:
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
(Both n1 ({n1}) and n2 ({n2}) <= 20) → Check Runs Test Table
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
