import numpy as np
import math
import pandas as pd
import html
import mgt2001


def BayesTHM(pre_probs, event='D'):
    """
Output: (Bayes):
>>> [num1, num2] >> meaning that given event B has occurred, the probability that the previous event occurred is num1
============
Input:
>>> BayesTHM(np.array[[0.9, 0.1], # being the previous conditions
                          [0.99, 0.9], # being the conditional probabilities: ([P(B | A), P(B | A̅)])
                          [0.01, 0.1]]) # being the complementaries of the conditional probabilities: ([P(B̅ | A), P(B̅ | A̅)])

The deafult number of columns is 2.

Pass in `event='B'` to specify the post event you are to compare.
    """
    events = [event, html.unescape('{}&#773;'.format(event))]
    n = pre_probs.shape[1]
    m = pre_probs.shape[0] - 1

    # prior_probs is an array representing the previous probability
    # cond_probs is a matrix representing the previous condition probability

    # only the first row being G and G_c respectively
    prior_probs = pre_probs[0, :]
    cond_probs = pre_probs[1:, :]

    # joint_probs is a matrix representing the joint probability
    # p_c_sums is an array representing the probability of prior event
    # Bayes_M_G is a matrix representing the post conditional probability

    joint_probs = np.zeros((m, n))
    p_c_sums = np.zeros(n)
    bayes = np.zeros((m, n))

    for i in range(m):
        joint_probs[i, ] = prior_probs * cond_probs[i, ]
        p_c_sums[i] = np.sum(joint_probs[i, ])
        bayes[i, ] = joint_probs[i, ] / p_c_sums[i]
        to_print = """The probability of post event {i}:
{p_c_sum:.7f}
================
The Bayes probability given the post event {i}:
{bayes}
        """.format(i=events[i], p_c_sum=p_c_sums[i], bayes=bayes[i, ])
        print(to_print)


def portfolio_analysis(stock_df, target_stocks, pss, ddof=0):
    """
    Return list of portfolios

    Usage:
    df = pd.read_excel('Xr07-TSE.xlsx')
    df = df.set_index(['Year', 'Month'])
    month_dict = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    df.rename(index=month_dict, inplace=True)

    target_stocks = ['BMO', 'MG', 'POW', 'RCL.B']
    pss = np.array([[.25, .25, .25, .25], [.2,.6, .1, .1], [.1, .2, .3, .4]])

    portfolios = portfolio_analysis(df[target_stocks], target_stocks, pss)

    --------------------
    cov_mat = np.cov(stock_df[target_stocks].values, rowvar=False) # [i, i] = variance of each stock
    cor_mat = np.corrcoef(df[target_stocks].values, rowvar = False)
    """

    NUMOFSTOCKS = len(target_stocks)
    NUMOFPORT = len(pss)

    cov_mat = np.cov(stock_df[target_stocks].values, rowvar=False, ddof=ddof)
    stock_des = pd.DataFrame(
        data=cov_mat, columns=target_stocks, index=target_stocks)

    stock_des.loc['Expected Returns'] = stock_df.mean()
    # stock_df.mean() # [mgt2001.geomean(stock_df.iloc[:, i]) for i in range(len(stock_df.columns))]

    portfolios = [stock_des.copy() for i in range(NUMOFPORT)]
    for i in range(NUMOFPORT):
        portfolios[i].loc['Weights'] = pss[i]
        portfolios[i]['Total'] = pd.Series(
            data=(portfolios[i].loc['Weights'].sum()), index=['Weights'])

    expected_values = [0 for i in range(NUMOFPORT)]
    var = [0 for i in range(NUMOFPORT)]
    std = list()

    for i in range(NUMOFPORT):
        expected_values[i] += (portfolios[i].loc['Weights']
                               * portfolios[i].loc['Expected Returns']).sum()

    for i in range(NUMOFPORT):
        var[i] = np.dot(portfolios[i].loc['Weights'][0:NUMOFSTOCKS].to_numpy(), np.dot(
            cov_mat, portfolios[i].loc['Weights'][0:NUMOFSTOCKS].to_numpy().transpose()))

    std = [math.sqrt(var[i]) for i in range(NUMOFPORT)]

    for i in range(NUMOFPORT):
        result = """======== Portfolio {i} Return ========
Expected Value:{tab}{tab}{exp:.7f}
Variance:{tab}{tab}{var:.7f}
Standard Deviation:{tab}{std:.7f}
        """.format(tab='\t', i=i+1, exp=expected_values[i], var=var[i], std=std[i])
        print(result)
    return portfolios


def covariance(x, mu_x, y, mu_y, prob):
    '''
    Usage: `covariance(ibm_x, ibm.expect(), ms_y, ms.expect(), prob)`
    Where `prob` equals 
        → x
        ↓ y

    Prob can be a two dimensional array or a `numpy` array.
    '''
    cov = 0
    for i, valy in enumerate(y):
        for j, valx in enumerate(x):
            cov += prob[i][j] * (valx - mu_x) * (valy - mu_y)
    return cov
