import pandas as pd
import numpy as np
import math
import scipy.stats as stats
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf


def f_oneway(data, treatment_name, value_name):
    results = smf.ols(f'{value_name} ~ C({treatment_name})', data=data).fit()
    aov_table = sms.anova_lm(results, typ=2)
    f_stat, p_value = aov_table['F'][0], aov_table['PR(>F)'][0]
    render_table = aov_table.copy()
    render_table.columns = ['Sum of Squares',
                            'Degree of Freedom', 'F', 'p-value']
    # render_table.index = ['Treatment', 'Error']
    render_table.loc['Total'] = render_table.sum()
    print(f'p-value: {p_value}')
    return aov_table, render_table, f_stat, p_value
