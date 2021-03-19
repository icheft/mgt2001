from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import scipy.stats as stats
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf


def shapiro(df, treatment_name_list, treatment_name, value_name):
    """
    check normal distribution
    """
    for i, name in enumerate(treatment_name_list):
        data = df[value_name][df[treatment_name] == name]
        stat, p = stats.shapiro(data)
        print(f'{i + 1}: Statistics={stat:.4f}, p={p:.4f}')
    return


def qq_plot(row, col, df, treatment_name_list, treatment_name, value_name, figsize=(8, 3), hspace=0.4, wspace=4):
    """
    check normal distribution
    """
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    for i, name in enumerate(treatment_name_list):
        ax = fig.add_subplot(row, col, i + 1)
        data = df[value_name][df[treatment_name] == name]
        sm.qqplot(data, stats.norm, fit=True, line='45', ax=ax)
        ax.set_title(treatment_name_list[i])

    fig.tight_layout()
    plt.show()
    return


def bartlett(df, treatment_name_list, treatment_name, value_name):
    """
    Equal Variances (barlett's Test)
    """
    data = []
    for i, name in enumerate(treatment_name_list):
        data.append(df[value_name][df[treatment_name] == name])
    stat = stats.bartlett(*data)
    print(f'p-value: {stat[1]}')
    return stat


def f_oneway(data, treatment_name, value_name):
    """
    return results, aov_table, render_table, f_stat, p_value
    """
    results = smf.ols(f'{value_name} ~ C({treatment_name})', data=data).fit()
    aov_table = sms.anova_lm(results, typ=2)
    f_stat, p_value = aov_table['F'][0], aov_table['PR(>F)'][0]
    render_table = aov_table.copy()
    render_table.columns = ['Sum of Squares',
                            'Degree of Freedom', 'F', 'p-value']
    # render_table.index = ['Treatment', 'Error']
    render_table.loc['Total'] = render_table.sum()
    print(f'p-value: {p_value}')
    return results, aov_table, render_table, f_stat, p_value


def f_random_block(data, treatment_name, block_name, value_name, precision=4):
    """
    return results, aov_table, render_table, treatment_f_stat, treatment_p_value, block_f_stat, block_p_value
    """
    results = smf.ols(
        f'{value_name} ~ C({treatment_name}) + C({block_name})', data=data).fit()
    aov_table = sms.anova_lm(results, typ=2)

    treatment_f_stat, treatment_p_value = aov_table['F'][0], aov_table['PR(>F)'][0]
    block_f_stat, block_p_value = aov_table['F'][1], aov_table['PR(>F)'][1]
    render_table = aov_table.copy()
    render_table.columns = ['Sum of Squares',
                            'Degree of Freedom', 'F', 'p-value']

    render_table.index = ['Treatment', 'Block', 'Error']

    render_table.loc['Total'] = render_table.sum()
    render_table.loc['Total', ['F', 'p-value']] = np.nan
    print(
        f'Treatment p-value (main): {treatment_p_value:.{precision}f}\nBlock p-value: {block_p_value:.{precision}f}')
    return results, aov_table, render_table, treatment_f_stat, treatment_p_value, block_f_stat, block_p_value


def f_twoway(data, factorA_name, factorB_name, value_name, precision=4):
    """
    return results, aov_table, render_table, factorA_f_stat, factorA_p_value, factorB_f_stat, factorB_p_value, inter_f_stat, inter_p_value
    """
    results = smf.ols(
        f'{value_name} ~ C({factorA_name}) + C({factorB_name}) + C({factorA_name}):C({factorB_name})', data=data).fit()
    aov_table = sms.anova_lm(results, typ=2)

    factorA_f_stat, factorA_p_value = aov_table['F'][0], aov_table['PR(>F)'][0]
    factorB_f_stat, factorB_p_value = aov_table['F'][1], aov_table['PR(>F)'][1]
    inter_f_stat, inter_p_value = aov_table['F'][2], aov_table['PR(>F)'][2]
    render_table = aov_table.copy()
    render_table.columns = ['Sum of Squares',
                            'Degree of Freedom', 'F', 'p-value']

    render_table.index = ['Factor A', 'Factor B', 'Interaction', 'Error']

    render_table.loc['Total'] = render_table.sum()
    render_table.loc['Total', ['F', 'p-value']] = np.nan
    print(
        f'Factor A\'s p-value: {factorA_p_value:.{precision}f}\nFactor B\'s p-value: {factorB_p_value:.{precision}f}\nInteraction p-value: {inter_p_value:.{precision}f}')
    return results, aov_table, render_table, factorA_f_stat, factorA_p_value, factorB_f_stat, factorB_p_value, inter_f_stat, inter_p_value
