from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as smm
import statsmodels.stats.outliers_influence as sso
import statsmodels
import statistics
import math
import time
import itertools
from scipy.optimize import curve_fit
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.api import acf, pacf, graphics

##################### MICHAEL #####################


def Outlier_and_InfObs(standard_resid=None, x_data=None, y_data=None, Multi=True, df=None):
    outlier_index = []
    infobs_index = []
    # print ('Outliers :')
    for i in range(len(standard_resid)):
        if (standard_resid[i] < -2 or standard_resid[i] > 2):
            outlier_index.append(i)
            # print (i, standard_resid[i])

    print("\n")

    if (not Multi):
        cov_mat = np.cov(y_data, x_data)
        x_bar = x_data.mean()
        nobs = len(x_data)
        h_val = 1 / nobs + (x_data - x_bar) ** 2 / (nobs - 1) / cov_mat[1, 1]
        # print(h_val)
        df1 = pd.DataFrame(h_val, columns=['hi'])
        filter = (df1['hi'] > 6 / nobs)
        print("\nInfluential Observations by hi :")
        print(df1['hi'].loc[filter])
    else:
        H = np.matmul(x_data, np.linalg.solve(
            np.matmul(x_data.T, x_data), x_data.T))
        df_w_h = df.copy().reset_index().rename(columns={'index': 'ID'})
        df_w_h['ID'] += 1
        df_w_h['h_ii'] = np.diagonal(H)
        # print (x_data.shape[1])
        k = x_data.shape[1]-1
        n = len(df_w_h['h_ii'])
        h_level = 3 * (k+1) / n
        # print("h_level = ", h_level)
        # print(" \n")
        for i in range(0, df_w_h.shape[0]):
            if df_w_h['h_ii'][i] > h_level:
                infobs_index.append(i)
        filter = (df_w_h['h_ii'] > h_level)
        # print("Influential Observations by hi = \n")
        # print(df_w_h['h_ii'].loc[filter])
        return outlier_index, infobs_index


##################### DEREK #####################
# Outliers DIY


def simple_outliers_DIY(df, xname, yname, alpha=0.05):
    # Fit regression model
    result = smf.ols(yname + '~' + xname, data=df).fit()

    # studentized residual
    st1, data1, ss3 = sso.summary_table(result, alpha=alpha)
    Residual = data1[:, 8]
    STD_Residual = data1[:, 10]
    mu = np.mean(STD_Residual)
    sigma = np.std(STD_Residual)

    print("◆ Outliers Finding\n")
    print("(remove by yourself!)\n")
    df_out = pd.DataFrame(STD_Residual, columns=['SD'])
    filter = (df_out['SD'] < -2) | (df_out['SD'] > 2)
    print("Outliers by SD = ")
    print(df_out['SD'].loc[filter])
    print("\nActual ID: ", df_out['SD'].loc[filter].index+1)
    return df_out['SD'].loc[filter].index

# compute p value from t statistics


def tpv(stat, dof, tail):
    if (tail == 'r'):
        return 1 - stats.t.cdf(stat, df=dof)  # right
    elif (tail == 'l'):
        return stats.t.cdf(stat, df=dof)  # left
    elif (tail == 'db'):
        if(stats.t.cdf(stat, df=dof) > 0.5):
            return 2 * (1 - stats.t.cdf(stat, df=dof))  # double
        else:
            return 2 * stats.t.cdf(stat, df=dof)  # double
    else:
        return -1  # error

# p value interpretation


def pvSig(pv):
    print("\n====== p value significance ======")
    if (pv <= 0.01):
        print(">>> highly sig, overwhelming evidence\n    sig, strong evidence\n    not sig, weak evidence\n    not sig, little to no evidence")
    elif (pv <= 0.05 and pv > 0.01):
        print("    highly sig, overwhelming evidence\n>>> sig, strong evidence\n    not sig, weak evidence\n    not sig, little to no evidence")
    elif (pv <= 0.1 and pv > 0.05):
        print("    highly sig, overwhelming evidence\n    sig, strong evidence\n>>> not sig, weak evidence\n    not sig, little to no evidence")
    elif (pv > 0.1):
        print("    highly sig, overwhelming evidence\n    sig, strong evidence\n    not sig, weak evidence\n>>> not sig, little to no evidence")
    else:
        print("BAD INPUT")
    print("===================================\n")

# r value interpretation


def rvInter(rv):
    print("\n====== R value interpretation ======")
    if (rv > 0):
        print("            [positive]")
    elif (rv < 0):
        print("            [negative]")
    else:
        print("        [no linear rellation]")
        return

    if (abs(rv) <= 0.25):
        print("    very strong\n    moderately strong\n    moderately weak\n>>> very weak")
    elif (abs(rv) <= 0.5 and abs(rv) > 0.25):
        print("    very strong\n    moderately strong\n>>> moderately weak\n    very weak")
    elif (abs(rv) <= 0.75 and abs(rv) > 0.5):
        print("    very strong\n>>> moderately strong\n    moderately weak\n    very weak")
    elif (abs(rv) <= 1 and abs(rv) > 0.75):
        print(">>> very strong\n    moderately strong\n    moderately weak\n    very weak")
    else:
        print("BAD INPUT")
    print("====================================\n")


def simple_regplot(df, xname, yname):
    _ = sns.regplot(x=xname, y=yname, data=df, color='b', ci=None)
    plt.title('Scatter Plot')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.show()


def simple_regmod(df, xname, yname):
    # Fit regression model
    result1 = smf.ols(yname + '~ ' + xname, data=df).fit()
    # Inspect the results
    print(result1.summary())

    b1_1 = result1.params[1]
    b0_1 = result1.params[0]
    print(f"Estimated model: y = {b0_1:.4f} + {b1_1:.4f} x")


def simple_durbin_watson(df, xname, yname, alpha=0.05):
    print("\n\n========== Durbin-Watson ==========\n")

    y_data = df[yname]
    x_data_ar = []
    x_data_ar = np.asarray(df[xname])

    x_data_T = x_data_ar.T
    x_data = pd.DataFrame({xname: x_data_T})
    x_data2 = sm.add_constant(x_data)
    olsmod = sm.OLS(y_data, x_data2)
    result = olsmod.fit()

    st, data, ss2 = sso.summary_table(result, alpha=alpha)
    print("Columns in data are: %s" % ss2)
    # Predicted value
    y_pre = data[:, 2]
    # Studentized Residual
    SD = data[:, 10]

    x_square_sum = np.vdot(SD, SD)
    print("x_square_sum = ", x_square_sum)
    size = SD.size
    print("size = ", size)
    x_d = np.zeros((size))
    print("x_d = ", x_d)
    l_size = size - 1
    for i in range(l_size):
        x_d[i + 1] = SD[i + 1] - SD[i]
    print("x_d = ", x_d)
    d = np.vdot(x_d, x_d) / x_square_sum
    print("d = ", d)


def chi2_normtest(stand_res, N, alpha=0.05):
    m = np.mean(stand_res)
    s = np.std(stand_res)
    prob_bins = np.zeros((N + 1))
    z_bins = np.zeros((N + 1))
    z_bins[0] = -4
    z_bins[N] = 4
    for i in range(0, N+1):
        prob_bins[i] = i/N
    for j in range(1, N):
        z_bins[j] = m + stats.norm.isf(1 - prob_bins[j]) * s
    counts, bins = np.histogram(stand_res, bins=z_bins)
    nobs = counts.sum()
    prob_e = np.zeros((N))
    for i in range(1, N+1):
        prob_e[i - 1] = prob_bins[i] - prob_bins[i - 1]
    freq_e = nobs * prob_e
    freq_o = counts
    if np.sum(freq_e < 5) > 0:
        print("Rule of five is not met.")
    else:
        chi_stat, pval = stats.chisquare(freq_o, freq_e)
        chi_pval = stats.chi2.sf(chi_stat, N - 3)
        print("Chi-squared test: statistics = %0.4f, p-value = %0.4f" %
              (chi_stat, chi_pval))
    df = freq_o.shape[0]-3
    crit_value = stats.chi2.ppf(1 - alpha, df)
    print("Critical value = %0.4f (defree of freedom = %d)" % (crit_value, df))

    return chi_pval


def runsTest(l, l_median):
    runs, n1, n2 = 1, 0, 0
    if(l[0]) >= l_median:
        n1 += 1
    else:
        n2 += 1
    # Checking for start of new run
    for i in range(len(l)):
        # no. of runs
        if (l[i] >= l_median and l[i-1] < l_median) or (l[i] < l_median and l[i-1] >= l_median):
            runs += 1
            # print(i, runs)
        # no. of positive values
        if(l[i]) >= l_median:
            n1 += 1
        # no. of negative values
        else:
            n2 += 1
    runs_exp = ((2*n1*n2)/(n1+n2)) + 1
    stan_dev = math.sqrt((2*n1*n2*(2*n1*n2-n1-n2))/(((n1+n2)**2)*(n1+n2-1)))
    z = (runs-runs_exp)/stan_dev
    pval_z = stats.norm.sf(abs(z)) * 2
    print('runs = ', runs)
    print('n1 = ', n1)
    print('n2 = ', n2)
    print('runs_exp = ', runs_exp)
    print('stan_dev = ', stan_dev)
    print('z = ', z)
    print('pval_z = ', pval_z)
    return pval_z


def simple_residual(df, xname, yname, alpha=0.05, resd_all=False, nobins=6):
    print("\n\n----------------------------\n|Residual Analysis - simple|\n----------------------------\n")
    print("using alpha = ", alpha)
    print("\n\n ◆ Residuals\n")

    # Fit regression model
    result = smf.ols(yname + '~' + xname, data=df).fit()

    # studentized residual
    st1, data1, ss3 = sso.summary_table(result, alpha=alpha)
    Residual = data1[:, 8]
    STD_Residual = data1[:, 10]
    mu = np.mean(STD_Residual)
    sigma = np.std(STD_Residual)

    if(resd_all == True):
        print("Original Residuals: \n", Residual, "\n")
        print("Standardized Residuals: \n", STD_Residual, "\n")
        print("mu:", mu)
        print("sigma:", sigma)
    else:
        print("mu:", mu)
        print("sigma:", sigma)

    # Normality Test
    print("\n\n ◆ Error Normality Test\n")
    print("H0: Errors are normally distributed.")
    print("H1: Errors are not normally distributed.")

    # Histogram
    print("\n\n   ◇ Histogram\n")
    counts, bins, patches = plt.hist(
        STD_Residual, nobins, density=False, facecolor='black', alpha=0.75)

    plt.xlabel('Standardized Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Standardized Residuals')
    plt.grid(True)
    bin_centers = [np.mean(k) for k in zip(bins[:-1], bins[1:])]
    plt.show()

    print(counts)
    print(bins)

    # Shapiro Test
    print("\n\n   ◇ Shapiro Test\n")
    stat, spv = stats.shapiro(STD_Residual)
    print(f"Statistics = {stat:.4f}, p-value = {spv:.4f}")
    pvSig(spv)

    # Chi^2 Test
    print("\n\n   ◇ Chi-squared Test\n")
    stand_res = STD_Residual
    N = nobins

    m = np.mean(stand_res)
    s = np.std(stand_res)
    prob_bins = np.zeros((N + 1))
    z_bins = np.zeros((N + 1))
    z_bins[0] = -4
    z_bins[N] = 4
    for i in range(0, N+1):
        prob_bins[i] = i/N
    for j in range(1, N):
        z_bins[j] = m + stats.norm.isf(1 - prob_bins[j]) * s
    counts, bins = np.histogram(stand_res, bins=z_bins)
    nobs = counts.sum()
    prob_e = np.zeros((N))
    for i in range(1, N+1):
        prob_e[i - 1] = prob_bins[i] - prob_bins[i - 1]
    freq_e = nobs * prob_e
    freq_o = counts
    if np.sum(freq_e < 5) > 0:
        print("Rule of five is not met.")
    else:
        chi_stat, pval = stats.chisquare(freq_o, freq_e)
        chi_pval = stats.chi2.sf(chi_stat, N - 3)
        print("Chi-squared test: statistics = %0.4f, p-value = %0.4f" %
              (chi_stat, chi_pval))
    df_fq = freq_o.shape[0]-3
    crit_value = stats.chi2.ppf(1 - alpha, df_fq)
    print("Critical value = %0.4f (defree of freedom = %d)" %
          (crit_value, df_fq))

    # pvSig(chi_pval)

    # Homoscedasticity and Heteroscedasticity
    print("\n\n ◆ Homoscedasticity and Heteroscedasticity\n")
    print("H_0: Randomness exists")
    print("H_1: Randomness doesn't exist")
    Id1 = data1[:, 0]
    plt.plot(Id1, STD_Residual, 'o', color='gray')
    plt.axhline(y=0, color='blue')
    plt.axhline(y=2, color='red')
    plt.axhline(y=-2, color='red')
    plt.title('Standardized Residual Plot')
    plt.xlabel('Observation No.')
    plt.ylabel('Standardized Residual')
    plt.show()

    # Dependence of the Error Variable
    print("\n\n ◆ Dependence of the Error Variable (Run Test)\n")
    print("H_0: Sample is random")
    print("H_1: Sample is not random")
    SD_median = statistics.median(STD_Residual)
    Z_pval = runsTest(STD_Residual, SD_median)
    print('p-value for run test z-statistic= ', Z_pval)
    pvSig(Z_pval)

    # Outliers
    print("\n\n ◆ Outliers Finding\n")
    print("(remove by yourself!)\n")
    df_out = pd.DataFrame(STD_Residual, columns=['SD'])
    filter = (df_out['SD'] < -2) | (df_out['SD'] > 2)
    print("Outliers by SD = ")
    print(df_out['SD'].loc[filter])
    print("\nActual ID: ", df_out['SD'].loc[filter].index+1)

    # Influential Observations
    print("\n\n ◆ Influential observations Finding\n")
    x_data = df[xname].values
    y_data = df[yname].values
    cov_mat1 = np.cov(y_data, x_data)
    x_data_bar = x_data.mean()
    data_nobs = len(x_data)
    h_val = 1 / data_nobs + (x_data - x_data_bar) ** 2 / \
        (data_nobs - 1) / cov_mat1[1, 1]
    # print(h_val)
    df_hi = pd.DataFrame(h_val, columns=['hi'])
    filter = (df_hi['hi'] > nobins / data_nobs)
    print("Influential Observations by hi = ", df_hi['hi'].loc[filter])
    print("\nAutal ID: ", df_hi['hi'].loc[filter].index+1)


def simple_modass(df, xname, yname, alpha=0.05, tail='db'):
    # Fit regression model
    result1 = smf.ols(yname + '~ ' + xname, data=df).fit()

    b1_1 = result1.params[1]
    b0_1 = result1.params[0]
    print(f"Estimated model: y = {b0_1:.4f} + {b1_1:.4f} x")

    print("\n\n---------------------------\n|     Model Assessing     |\n---------------------------\n")
    print("using alpha = ", alpha)

    print("\n\n ◆ Standard Error of Estimate\n")
    s2_e = result1.mse_resid
    print(f"MSE = {s2_e:f}")
    s_e = result1.mse_resid ** 0.5
    print(f"Standard errors = {s_e:f}")
    y_bar = df[yname].mean()
    print(f"y mean = {y_bar:.4f}")
    print(
        f"The absolute value of standard errors is about {abs(s_e/y_bar)*100:.0f}% of mean of independent variables.\n")

    print("\n\n ◆ Coefficient of Determination\n")
    R2 = result1.rsquared
    print(f"R^2 = {R2:f}")
    R = np.sign(b1_1) * R2 ** 0.5
    print(f"R = {R:f}")

    print(
        f"\nR^2 value interpretation\nAbout {R2*100:.0f}% of the variation in the dependent variables is explained by independent ones, the rest remains unexplained.")
    rvInter(R)

    print("\n\n ◆ Studetn-t test for beta1(slope)\n")
    dof = len(df) - 2
    tv = R * ((dof - 2)/(1 - R ** 2)) ** 0.5
    LCL = stats.t.ppf(alpha / 2, dof - 2)
    UCL = stats.t.ppf(1 - alpha / 2, dof - 2)
    print('t = ', tv)
    print('t_LCL = ', LCL)
    print('t_UCL = ', UCL)

    print(f"\np-value of t-stat tail: {tail}")
    tp = tpv(tv, dof, tail)
    print("p-value of t test = ", tp)
    pvSig(tp)

    print("\n\n ◆ Coefficient of Correlation\n")
    cor_mat = np.corrcoef(df[[xname, yname]].values, rowvar=False)
    n = df.shape[0]
    r = cor_mat[1, 0]
    tv_cc = r * ((n-2)/(1 - r**2)) ** 0.5
    t_critical = stats.t.ppf(0.975, n - 2)
    pval = stats.t.sf(np.abs(tv_cc), n - 2)*2

    print('r = ', r)
    print('t_critical = ', t_critical)
    print('t = ', tv_cc)
    print('p_value = ', pval)


def simple_CIPIPRE(x, y, x1, alpha=0.05):
    print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n|CI PI for simple regression|\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    print("using alpha = ", alpha)

    x_new = np.array([1, x1])
    print("make Confidence Interval and Prediction Interval predictions at mean of x = ", x1)
    x2 = sm.add_constant(x)
    olsmod = sm.OLS(y, x2)
    result_reg = olsmod.fit()
    y_head = np.dot(result_reg.params, x_new)
    print("y_head = ", y_head)
    (t_minus, t_plus) = stats.t.interval(
        alpha=(1.0 - alpha), df=result_reg.df_resid)
    cov_mat1 = np.cov(y, x)
    x_bar = x.mean()
    core1 = (1 / result_reg.nobs + (x1 - x_bar) ** 2 /
             (result_reg.nobs - 1) / cov_mat1[1, 1]) ** 0.5
    core2 = (1 + 1 / result_reg.nobs + (x1 - x_bar) ** 2 /
             (result_reg.nobs - 1) / cov_mat1[1, 1]) ** 0.5
    lower_bound = y_head + t_minus * (result_reg.mse_resid ** 0.5) * core1
    upper_bound = y_head + t_plus * (result_reg.mse_resid ** 0.5) * core1
    half_interval = t_plus * (result_reg.mse_resid ** 0.5) * core1
    lower_bound2 = y_head + t_minus * (result_reg.mse_resid ** 0.5) * core2
    upper_bound2 = y_head + t_plus * (result_reg.mse_resid ** 0.5) * core2
    half_interval2 = t_plus * (result_reg.mse_resid ** 0.5) * core2

    print(
        f"\n{100*(1-alpha):.0f}% confidence interval for mean: [{lower_bound:.4f}, {upper_bound:.4f}], or {y_head:.4f} +- {half_interval:.4f}")
    print(
        f"\n{100*(1-alpha):.0f}% prediction interval: [{lower_bound2:.4f}, {upper_bound2:.4f}], or {y_head:.4f} +- {half_interval2:.4f}")


def simple_CIPIINT_regplot(df, xname, yname, alpha=0.05):
    print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n|CI PI Interval plot - simple|\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    print("using alpha = ", alpha)

    df_sorted = df.sort_values([xname])
    result = smf.ols(yname + '~' + xname, data=df_sorted).fit()
    x = df_sorted[xname].values
    y = df_sorted[yname].values
    st, data, ss2 = sso.summary_table(result, alpha=alpha)
    fittedvalues = data[:, 2]
    predict_mean_se = data[:, 3]
    predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T
    predict_ci_low, predict_ci_upp = data[:, 6:8].T

    plt.plot(x, y, 'o', color='gray')
    plt.plot(x, fittedvalues, '-', lw=0.5)
    plt.plot(x, predict_mean_ci_low, 'r-', lw=0.4)
    plt.plot(x, predict_mean_ci_upp, 'r-', lw=0.4)
    plt.plot(x, predict_ci_low, 'b--', lw=0.4)
    plt.plot(x, predict_ci_upp, 'b--', lw=0.4)
    plt.title('CI PI plot')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.legend(['data points', 'regression model', 'confidence interval',
                'prediction interval'], title='Legends', bbox_to_anchor=(1.3, 1), prop={'size': 6})
    plt.show()


def simple(step, df, xname, yname, alpha=0.05, tail='db', nobins=6, resd_all=False):
    if step == 1:
        simple_regplot(df, xname, yname)
    elif step == 2:
        print("\npropose a statistical model\n")
    elif step == 3:
        simple_regmod(df, xname, yname)
    elif step == 4:
        print("\nfor autocorrelation and others, please determine by yourself!\n")
        simple_durbin_watson(df, xname, yname, alpha=alpha)
    elif step == 5:
        print("\nremember to remove outliers or do some modifications.\n")
        simple_residual(df, xname, yname, alpha=alpha,
                        resd_all=resd_all, nobins=nobins)
    elif step == 6:
        simple_modass(df, xname, yname, alpha=alpha, tail=tail)
    elif step == 7:
        print("\ninterpretation\n")
    elif step == 8:
        print(
            "\nsimple_CIPIPRE(df[xname], df[yname], x_input...) won't run here\n")
        simple_CIPIINT_regplot(df, xname, yname, alpha=alpha)
    else:
        print("\nbad input for step!\n")


def multiple_regplot(df, xnames, yname):
    for aname in xnames:
        x_var = df[aname].values
        _ = sns.regplot(x=x_var, y=df[yname].values,
                        data=df, color='b', ci=None)
        plt.xlabel(aname)
        plt.ylabel(yname)
        plt.show()

    df_ = df[[yname] + xnames]
    corr1 = df_.corr()
    corr1
    _ = sns.heatmap(corr1, annot=True)


def multiple_modpropose(xnames, yname):
    print("\n\n---------- Model Proposal ----------\n")
    print("Model proposal,<br>")

    mod = "$y = \\beta_0 + "
    for i in range(len(xnames)):
        coe = "\\beta_" + str(i+1) + "x_" + str(i+1) + " + "
        mod = mod + coe
    mod = mod + "\\epsilon$<br>"
    print(mod)

    print("where y is ", yname, "<br>")

    exp = "and "
    for j in range(len(xnames)):
        xexp = "$x_" + str(j+1) + "$ is " + xnames[j] + ", "
        exp = exp + xexp

    print(exp)


def multiple_regmod(df, xnames, yname, alpha=0.05):
    y_data = df[yname]
    x_data_ar = []
    for i in range(len(xnames)):
        x_data_ar.append(df[xnames[i]])
    x_data_ar = np.asarray(x_data_ar)

    x_data_T = x_data_ar.T
    x_data = pd.DataFrame(x_data_T, columns=xnames)
    x_data2 = sm.add_constant(x_data)
    olsmod = sm.OLS(y_data, x_data2)
    result = olsmod.fit()
    print(f"\n\nusing alpha = {alpha:.2f}")
    print(result.summary())

    print("\nEstimated model: y = %0.4f" % (result.params[0]), end=' ')
    for c, x in zip(result.params[1:], list(range(1, len(xnames)+1))):
        print('+', end='') if c > 0 else print('-', end='')
        print(" %0.4f x%d " % (abs(c), x), end='')


def multiple_durbin_watson(df, xnames, yname, alpha=0.05):
    print("\n\n========== Durbin-Watson ==========\n")

    y_data = df[yname]
    x_data_ar = []
    for i in range(len(xnames)):
        x_data_ar.append(df[xnames[i]])
    x_data_ar = np.asarray(x_data_ar)

    x_data_T = x_data_ar.T
    x_data = pd.DataFrame(x_data_T, columns=xnames)
    x_data2 = sm.add_constant(x_data)
    olsmod = sm.OLS(y_data, x_data2)
    result = olsmod.fit()

    st, data, ss2 = sso.summary_table(result, alpha=alpha)
    print("Columns in data are: %s" % ss2)
    # Predicted value
    y_pre = data[:, 2]
    # Studentized Residual
    SD = data[:, 10]

    x_square_sum = np.vdot(SD, SD)
    print("x_square_sum = ", x_square_sum)
    size = SD.size
    print("size = ", size)
    x_d = np.zeros((size))
    print("x_d = ", x_d)
    l_size = size - 1
    for i in range(l_size):
        x_d[i + 1] = SD[i + 1] - SD[i]
    print("x_d = ", x_d)
    d = np.vdot(x_d, x_d) / x_square_sum
    print("d = ", d)


def multiple_residual(df, xnames, yname, alpha=0.05, nobins=6):
    print("\n\n----------------------------\n|Residual Analysis - multiple|\n----------------------------\n")
    print("using alpha = ", alpha)
    print("\n\n ◆ Residuals\n")

    y_data = df[yname]
    x_data_ar = []
    for i in range(len(xnames)):
        x_data_ar.append(df[xnames[i]])
    x_data_ar = np.asarray(x_data_ar)

    x_data_T = x_data_ar.T
    x_data = pd.DataFrame(x_data_T, columns=xnames)
    x_data2 = sm.add_constant(x_data)
    olsmod = sm.OLS(y_data, x_data2)
    result = olsmod.fit()

    st, data, ss2 = sso.summary_table(result, alpha=alpha)
    print("Columns in data are: %s" % ss2)
    # Predicted value
    y_pre = data[:, 2]
    # Studentized Residual
    SD = data[:, 10]

    mu = np.mean(SD)
    sigma = np.std(SD)

    # Normality Test
    print("\n\n ◆ Error Normality Test\n")
    print("H0: Errors are normally distributed.")
    print("H1: Errors are not normally distributed.")

    # Histogram
    print("\n\n   ◇ Histogram\n")
    fig, ax = plt.subplots()
    counts, bins, patches = plt.hist(
        SD, nobins, density=False, facecolor='g', alpha=0.75)
    plt.xlabel('Standardized Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Standardized Residuals_Car Prices')
    plt.grid(True)
    bin_centers = [np.mean(k) for k in zip(bins[:-1], bins[1:])]
    plt.show()

    print(counts)
    print(bins)

    # qqplot
    print("\n\n   ◇ QQ-plot\n")
    fig = sm.qqplot(SD, stats.norm, fit=True, line='45')
    plt.show()
    print()

    # Shapiro Test
    print("\n\n   ◇ Shapiro Test\n")
    stat, spv = stats.shapiro(SD)
    print(f"Statistics = {stat:.4f}, p-value = {spv:.4f}")
    pvSig(spv)

    # Chi^2 Test
    print("\n\n   ◇ Chi-squared Test\n")
    stand_res = SD
    N = nobins - 1

    m = np.mean(stand_res)
    s = np.std(stand_res)
    prob_bins = np.zeros((N + 1))
    z_bins = np.zeros((N + 1))
    z_bins[0] = -4
    z_bins[N] = 4
    for i in range(0, N+1):
        prob_bins[i] = i/N
    for j in range(1, N):
        z_bins[j] = m + stats.norm.isf(1 - prob_bins[j]) * s
    counts, bins = np.histogram(stand_res, bins=z_bins)
    nobs = counts.sum()
    prob_e = np.zeros((N))
    for i in range(1, N+1):
        prob_e[i - 1] = prob_bins[i] - prob_bins[i - 1]
    freq_e = nobs * prob_e
    freq_o = counts
    if np.sum(freq_e < 5) > 0:
        print("Rule of five is not met.")
    else:
        chi_stat, pval = stats.chisquare(freq_o, freq_e)
        chi_pval = stats.chi2.sf(chi_stat, N - 3)
        print("Chi-squared test: statistics = %0.4f, p-value = %0.4f" %
              (chi_stat, chi_pval))
    df_fq = freq_o.shape[0]-3
    crit_value = stats.chi2.ppf(1 - alpha, df_fq)
    print("Critical value = %0.4f (defree of freedom = %d)" %
          (crit_value, df_fq))

    # pvSig(chi_pval)

    # Homoscedasticity and Heteroscedasticity
    print("\n\n ◆ Homoscedasticity and Heteroscedasticity\n")
    print("H_0:Randomness exists")
    print("H_0:Randomness doesn't exist")

    st, data, ss2 = sso.summary_table(result, alpha=alpha)
    print("\nColumns in data are: %s" % ss2)
    # Predicted value
    y_pre = data[:, 2]
    # Studentized Residual
    SD = data[:, 10]

    plt.plot(y_pre, SD, 'o', color='gray')
    plt.axhline(y=2, color='red', lw=0.8)
    plt.axhline(y=0, color='blue')
    plt.axhline(y=-2, color='red', lw=0.8)
    plt.title('Standardized Residual Plot')
    plt.xlabel('Predicted y value')
    plt.ylabel('Standardized Residual')
    plt.show()

    # autocorrelation
    # Dependence of the Error Variable
    print("\n\n ◆ Dependence of the Error Variable (Run Test)\n")
    print("H_0: Sample is random")
    print("H_1: Sample is not random")

    print("\nColumns in data are: %s" % ss2)

    Id1 = data[:, 0]
    plt.plot(Id1, SD, 'o', color='gray')
    plt.axhline(y=0, color='blue')
    plt.axhline(y=2, color='red')
    plt.axhline(y=-2, color='red')
    plt.title('Standardized Residual Plot')
    plt.xlabel('Observation No.')
    plt.ylabel('Standardized Residual')
    plt.show()

    SD_median = statistics.median(SD)
    Z_pval = runsTest(SD, SD_median)
    print('p_value for Z-statistic= ', Z_pval)

    pvSig(Z_pval)

    # Outliers
    print("\n\n ◆ Outliers Finding\n")
    df_out = pd.DataFrame(SD, columns=['SD'])
    filter = (df_out['SD'] < -2) | (df_out['SD'] > 2)
    print("Outliers by SD = ")
    print(df_out['SD'].loc[filter])
    print("\nActual ID: ", df_out['SD'].loc[filter].index+1)

    # Influential Observations by hii
    print("\n\n ◆ Influential observations Finding by hii\n")
    x_data2 = np.array(x_data2)
    H = np.matmul(x_data2, np.linalg.solve(
        np.matmul(x_data2.T, x_data2), x_data2.T))
    df['hii'] = np.diagonal(H)
    df_1h = pd.DataFrame(df['hii'])
    k = result.df_model
    n = len(df_1h['hii'])
    h_level = 3 * (k+1) / n
    print("h_level = ", h_level)
    filter = (df_1h['hii'] > h_level)
    print("\nInfluential Observations by hi =\n")
    print(df_1h['hii'].loc[filter])

    # Influential Observations by Cook's Distance
    print("\n\n ◆ Influential observations Finding by Cook's Distance\n")
    s2_e = result.mse_resid
    k = result.df_model
    y_a = data[:, 1]
    y_f = data[:, 2]
    h_i = df['hii']
    CD_arr = np.square(y_a - y_f) / s2_e / (k - 1) * h_i / np.square(1 - h_i)
    CD = np.array(CD_arr)
    df_cd = pd.DataFrame(CD, columns=['CD'])
    print(df_cd.head())
    filter = (df_cd['CD'] > 1)
    print("Influential Observations by Cook's Distances =\n")
    print(df_cd['CD'].loc[filter])


def multiple_modass(df, xnames, yname, alpha=0.05):
    y_data = df[yname]
    x_data_ar = []
    for i in range(len(xnames)):
        x_data_ar.append(df[xnames[i]])
    x_data_ar = np.asarray(x_data_ar)

    x_data_T = x_data_ar.T
    x_data = pd.DataFrame(x_data_T, columns=xnames)
    x_data2 = sm.add_constant(x_data)
    olsmod = sm.OLS(y_data, x_data2)
    result = olsmod.fit()

    print("\n\n---------------------------\n|     Model Assessing     |\n---------------------------\n")
    print("using alpha = ", alpha)

    print("\n\n ◆ Standard Error of Estimate\n")
    s2_e = result.mse_resid
    print(f"MSE = {s2_e:f}")
    s_e = result.mse_resid ** 0.5
    print("Standard error = ", s_e)
    y_bar = df[yname].mean()
    print("y mean = ", y_bar)
    print("y STD = ", df[yname].std())
    print(
        f"The absolute value of standard errors is about {abs(s_e/y_bar)*100:.0f}% of mean of independent variables.\n")

    R2 = result.rsquared
    print("\nCoefficient of Determination")
    print("R^2 = ", result.rsquared)
    print("Adjusted R^2 = ", result.rsquared_adj)

    print(
        f"\nR^2 value interpretation\nAbout {R2*100:.0f}% of the variation in the dependent variables is explained by the model, the rest remains unexplained.")
    rvInter(R2**0.5)

    print("\n\n ◆ Over-fitting?\n")
    diffrra = abs(result.rsquared - result.rsquared_adj)
    print("|R^2 - Ra^2| = ", diffrra)
    if(diffrra > 0.06):
        print("|R^2 - Ra^2| >= 0.06 indicating that the model has the problem of over-fitting.")
    else:
        print("|R^2 - Ra^2| < 0.06 indicating that the model doesn't have the problem of over-fitting.")

    print("\n\n ◆ F-test of ANOVA\n")
    print("Testing hypothesis,")
    print("H_0: \beta_1 = \beta_2 = \dots = \beta_n = 0<br>")
    print("H_1: \text{at least one } \beta_i \neq 0")

    f_res = result.fvalue
    MSE = result.mse_resid
    df_model = result.df_model
    df_error = result.df_resid
    MSR = f_res * MSE
    SSR = MSR * df_model
    print("SSR = ", SSR, "\tdf = ", df_model, "\tMSR = ", MSR)
    print("SSE = ", MSE * df_error, "\tdf = ", df_error, "\tMSE = ", MSE)
    print("F = MSR / MSE = ", MSR / MSE)
    fpv = result.f_pvalue
    print("F p-value = ", fpv)

    pvSig(fpv)


def multiple_CIPIPRE_(xdata, yval, x1, alpha=0.05):
    print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n|CI PI for simple regression|\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    print("using alpha = ", alpha)

    print("To make Confidence Interval and Prediction Interval prediction at mean of x = ", x1)
    x_data_T = xdata.T
    x_data2 = sm.add_constant(x_data_T)
    olsmod = sm.OLS(yval, x_data2)
    result_reg = olsmod.fit()
    y_head = np.dot(result_reg.params, x1)
    print("y_head = ", y_head)
    (t_minus, t_plus) = stats.t.interval(
        alpha=(1.0 - alpha), df=result_reg.df_resid)
    core1 = (result_reg.mse_resid * np.matmul(x1,
                                              np.linalg.solve(np.matmul(x_data2.T, x_data2), x1))) ** 0.5
    lower_bound = y_head + t_minus * core1
    upper_bound = y_head + t_plus * core1
    core2 = (result_reg.mse_resid * (1 + np.matmul(x1,
                                                   np.linalg.solve(np.matmul(x_data2.T, x_data2), x1)))) ** 0.5
    lower_bound2 = y_head + t_minus * core2
    upper_bound2 = y_head + t_plus * core2

    print(
        f"\n{100*(1-alpha):.0f}% confidence interval for mean: [{lower_bound:.4f}, {upper_bound:.4f}]")
    print(
        f"\n{100*(1-alpha):.0f}% prediction interval: [{lower_bound2:.4f}, {upper_bound2:.4f}]")


def multiple_CIPIPRE(df, xnames, yname, xx, alpha=0.05):
    x0 = [1]
    x1 = x0 + xx

    yval = df[yname]
    xdata_ar = []

    for i in range(len(xnames)):
        xdata_ar.append(df[xnames[i]])
    xdata_ar = np.asarray(xdata_ar)

    x1_ = np.array(x1)

    multiple_CIPIPRE_(xdata_ar, yval, x1_, alpha)


def multiple(step, df, xnames, yname, alpha=0.05, tail='db', nobins=6):
    if step == 1:
        multiple_regplot(df, xnames, yname)
    elif step == 2:
        multiple_modpropose(xnames, yname)
    elif step == 3:
        multiple_regmod(df, xnames, yname)
    elif step == 4:
        print("\nfor autocorrelation and others, please determine by yourself!\n")
        multiple_durbin_watson(df, xnames, yname, alpha=alpha)
    elif step == 5:
        print("\nremember to remove outliers or do some modifications.\n")
        multiple_residual(df, xnames, yname, alpha=alpha, nobins=nobins)
    elif step == 6:
        multiple_modass(df, xnames, yname, alpha=alpha)
    elif step == 7:
        print("\ninterpretation\n")
    elif step == 8:
        print("\multiple_CIPIPRE (df, xnames, yname, xx...) won't run here\n")
    else:
        print("\nbad input for step!\n")


def time_add(df, name='Time'):
    time = []
    for i in range(df.shape[0]):
        time.append(i)
    print(time)

    df[name] = time
    return df


def value_map_ln(df, target):
    lnv = []
    for i in range(df.shape[0]):
        lnv.append(math.log((df[target].values[i])))

    newname = "ln_" + target
    df[newname] = lnv
    return df


def outliers_rm(df, out):
    df = df.drop(df.index[out])
    df = df.reset_index()
    return df
