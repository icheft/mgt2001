import statistics

# compute p value from t statistics
def tpv (stat, dof, tail):
    if (tail == 'r'):
        return 1 - stats.t.cdf(stat, df = dof) #right
    elif (tail == 'l'):
        return stats.t.cdf(stat, df = dof) #left
    elif (tail == 'db'):
        if(stats.t.cdf(stat, df = dof) > 0.5):
            return 2 * (1 - stats.t.cdf(stat, df = dof)) # double
        else:
            return 2 * stats.t.cdf(stat, df = dof) # double
    else:
        return -1 # error

# p value interpretation
def pvSig (pv):
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
def rvInter (rv):
    print("\n====== R value interpretation ======")
    if (rv > 0):
        print("            [positive]")
    elif (rv <0):
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

# simple regression
def simple_regmod(df, xname, yname, alpha = 0.05, ass = False, tail = 'db'):
    # tail is for coefficient of determination

    # Fit regression model 
    result1 = smf.ols(yname + '~ ' + xname, data = df).fit()
    # Inspect the results
    print(f"using alpha = {alpha:.2f}")
    print(result1.summary())

    b1_1 = result1.params[1]
    b0_1 = result1.params[0]
    print(f"Estimated model: y = {b0_1:.4f} + {b1_1:.4f} x")
    
    if(ass == True):
        print("\n\n---------------------------\n|     Model Assessing     |\n---------------------------\n")

        s2_e = result1.mse_resid
        print(f"MSE = {s2_e:f}")
        s_e = result1.mse_resid ** 0.5
        print(f"Standard errors = {s_e:f}")
        y_bar = df[yname].mean()
        print(f"y mean = {y_bar:.4f}")
        print(f"The absolute value of standard errors is about {abs(s_e/y_bar)*100:.0f}% of mean of independent variables.\n")

        print("\nCoefficient of Determination")

        R2 = result1.rsquared
        print(f"R^2 = {R2:f}")
        R = np.sign(b1_1) * R2 ** 0.5
        print(f"R = {R:f}")

        print(f"\nR^2 value interpretation\nAbout {R2*100:.0f}% of the variation in the dependent variables is explained by independent ones, the rest remains unexplained.")
        rvInter(R)

        dof = len(df) - 2
        tv = R * ((dof - 2)/(1 - R ** 2)) ** 0.5
        LCL = stats.t.ppf(alpha / 2, dof - 2)
        UCL = stats.t.ppf(1 - alpha / 2, dof - 2)
        print('t = ', tv)
        print('t_LCL = ', LCL)
        print('t_UCL = ', UCL)

        print(f"\np-value of t-stat tail: {tail}")
        tp = tpv(tv, dof, tail)
        print(tp)

        pvSig(tp)

# CI PI plot for simple regression
def CIPIInt_simple_regplot(df, xname, yname, alpha = 0.05):
    df_sorted = df.sort_values([xname])
    result = smf.ols(yname + '~' + xname, data = df_sorted).fit()
    x = df_sorted[xname].values
    y = df_sorted[yname].values
    st, data, ss2 = sso.summary_table(result, alpha = alpha)
    fittedvalues = data[:, 2]
    predict_mean_se  = data[:, 3]
    predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T
    predict_ci_low, predict_ci_upp = data[:, 6:8].T
    
    plt.plot(x, y, 'o', color = 'gray')
    plt.plot(x, fittedvalues, '-', lw=0.5)
    plt.plot(x, predict_mean_ci_low, 'r-', lw=0.4)
    plt.plot(x, predict_mean_ci_upp, 'r-', lw=0.4)
    plt.plot(x, predict_ci_low, 'b--', lw=0.4)
    plt.plot(x, predict_ci_upp, 'b--', lw=0.4)
    plt.title('CI PI plot')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.legend(['data points', 'regression model', 'confidence interval', 'prediction interval'], title = 'Legends', bbox_to_anchor = (1.3, 1), prop={'size': 6})
    plt.show()
    
# chi-squared normality test
def chi2_normtest (stand_res, N, alpha = 0.05): 
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
        print("Chi-squared test: statistics = %0.4f, p-value = %0.4f" % (chi_stat, chi_pval))
    df = freq_o.shape[0]-3
    crit_value = stats.chi2.ppf(1 - alpha, df)
    print("Critical value = %0.4f (defree of freedom = %d)" % (crit_value, df))
    
    return chi_pval

# residual analysis for simple resgression
def simple_residual(df, xname, yname, alpha = 0.05, resd_all = True, nobins = 6):

    # Fit regression model 
    result = smf.ols(yname + '~' + xname, data = df).fit()

    # studentized residual
    st1, data1, ss3 = sso.summary_table(result, alpha = alpha)
    Residual = data1[:, 8]
    STD_Residual = data1[:,10]
    mu = np.mean(STD_Residual)
    sigma = np.std(STD_Residual)
    
    if(resd_all == True):
        print("Residuals: \n", Residual, "\n")
        print("Standardized Residuals: \n", STD_Residual, "\n")
        print("mu:", mu)
        print("sigma:", sigma)
    else:
        print("mu:", mu)
        print("sigma:", sigma)
    
    ### H0: normally distributed.
    ### H1: not normally distributed.
    
    ## Histogram
    print("\nHistogram")
    counts, bins, patches = plt.hist(STD_Residual, nobins, density = False, facecolor = 'black', alpha = 0.75)

    plt.xlabel('Standardized Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Standardized Residuals')
    plt.grid(True)
    bin_centers = [np.mean(k) for k in zip(bins[:-1], bins[1:])]
    plt.show()

    print(counts)
    print(bins)
    
    # Shapiro Test
    print('\nShapiro test')
    stat, spv = stats.shapiro(STD_Residual)
    print(f"Statistics = {stat:.4f}, p-value = {spv:.4f}")
    pvSig(spv)
    
    # Chi^2 Test
    print('\nChi-squared test')
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
        print("Chi-squared test: statistics = %0.4f, p-value = %0.4f" % (chi_stat, chi_pval))
    df = freq_o.shape[0]-3
    crit_value = stats.chi2.ppf(1 - alpha, df)
    print("Critical value = %0.4f (defree of freedom = %d)" % (crit_value, df))

    pvSig(chi_pval)
    
    '''
    # Outliers
    print("\nOutliers finding")
    df_out = pd.DataFrame(STD_Residual, columns = ['SD'])
    filter = (df_out['SD'] < -2) | (df_out['SD'] > 2)
    print("Outliers by SD = ")
    print(df_out['SD'].loc[filter])
    print("\nActual ID: ", df_out['SD'].loc[filter].index+1)
    
    # Influential Observations
    print("\nInfluential observations finding")
    x_data = df[xname].values
    y_data = df[yname].values
    cov_mat1 = np.cov(y_data, x_data)
    x_data_bar = x_data.mean()
    data_nobs = len(x_data)
    h_val = 1 / data_nobs + (x_data - x_data_bar) ** 2 / (data_nobs - 1) / cov_mat1[1,1]
    print(h_val)
    df_hi = pd.DataFrame(h_val, columns = ['hi'])
    filter = (df_hi['hi'] > nobins / data_nobs )
    print("Influential Observations by hi = ", df_hi['hi'].loc[filter])
    print("\nAutal ID: ", df_hi['hi'].loc[filter].index+1)
    '''

# scedasticity plot for simple regression
def simple_scedasticity_plot(df, xname, yname, alpha = 0.05, nofig = 2):
    result = smf.ols(yname + '~' + xname, data = df).fit()
    st1, data1, ss3 = sso.summary_table(result, alpha = alpha)
    Residual = data1[:, 8]
    STD_Residual = data1[:,10]
    
    if(nofig == 2):
        y_pre = data1[:, 2]
        plt.plot(y_pre, Residual, 'o', color = 'gray')
        plt.axhline(y=0, color = 'blue')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Employment')
        plt.ylabel('Residual')
        plt.show()
    
    y_pre = data1[:, 2]
    plt.plot(y_pre, STD_Residual, 'o', color = 'gray')
    plt.axhline(y=0, color = 'blue')
    plt.axhline(y=2, color = 'red')
    plt.axhline(y=-2, color = 'red')
    plt.title('Standardized Residual Plot')
    plt.xlabel('Predicted Employment')
    plt.ylabel('Standardized Residual')
    plt.show()

# Runs test
def runsTest(l, l_median): 
    runs, n1, n2 = 1, 0, 0
    # Checking for start of new run
    if(l[0]) >= l_median:
        n1 += 1   
    else:
        n2 += 1   
    for i in range(1, len(l)): 
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

# Dependence of the Error Variable
def simple_randomness_test(df, xname, yname, alpha = 0.05):

    result = smf.ols(yname + '~' + xname, data = df).fit()
    st1, data1, ss3 = sso.summary_table(result, alpha = alpha)
    Residual = data1[:, 8]
    STD_Residual = data1[:,10]
    
    Id = data1[:, 0]
    plt.plot(Id, STD_Residual, 'o', color = 'gray')
    plt.axhline(y=0, color = 'blue')
    plt.axhline(y=2, color = 'red')
    plt.axhline(y=-2, color = 'red')
    plt.title('Standardized Residual Plot')
    plt.xlabel('Observation No.')
    plt.ylabel('Standardized Residual')
    plt.show()

    STD_Residual_median = statistics.median(STD_Residual)
    Z_pval = runsTest(STD_Residual, STD_Residual_median) 
    print('p-value for z-statistic = ', Z_pval)
    
    pvSig(Z_pval)