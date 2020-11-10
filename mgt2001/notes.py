def measure_of_movement():
    """
    m1 : 1st moment Σx/n ~= mean or x_bar
    m2 : 2nd moment Σ(x - x_bar)^2/n ~ Variance (for Population)
    m3 : 3rd moment Σ(x - x_bar)^3/n
    m4 : 4th moment Σ(x - x_bar)^4/n
    """


def kurtosis():
    """
    Please refer to measure_of_movement.__doc__ for ms
    ---
    K = m4 / (m2 ^ 2)
    """


def skewness():
    """
    Please refer to measure_of_movement.__doc__ for ms
    ------------
    G = m3 / sqrt(m2 ^ 3)
    ---
    A measure of a data set's deviation from symmetry:
    + G > 0: Skewed to right
    + G = 0: Symmetric
    + G < 0: Skewed to left

    """


def color_list():
    """
    'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'
    """


def README():
    """
    # TL;DR
    List of mgt2001 functions (from mgt2001 import *):
    + .
        + geomean(rate)
        + ori_autolabel(rects)
        + autolabel(rects, **kwargs)
        + color_palette(n, cmap='jet')
        + add_margin(ax, x=0.05, y=.05)
        + Pos()
            + Pos().width
            + Pos().returnPos()
            + Pos().restructure
    + per
        + per.percentile(data, percentage)
        + per.percentrank(data, value)
    + des
        + des.outlier(data, base=per) --> per = mgt.2001 per
        + des.kurtosis(df)
        + des.convert_excel_kurtosis(K, n)
        + des.skew(df)
        + des.convert_excel_skew(G, n)
    + prob
        + prob.BayesTHM(pre_probs, event='D')
        + prob.portfolio_analysis(stock_df, target_stocks, pass)
    + notes
        + notes.measure_of_movement.__doc__
        + notes.kurtosis.__doc__
        + notes.skewness.__doc__
        + notes.color_list.__doc__
        + notes.README.__doc__

    ## `__init__.py`
    + `geomean(rate)`: return the geometric mean, rate being np.array()
    + `ori_autolabel(rects, truncate='{}')`: attach a text label above each bar in *rects*, truncate can be filled in like '{:.5f}'
    + `autolabel(rects, **kwargs)`: 
        + list of kwargs:
            + bool original
            + string truncate: in the '{}' format. Can be filled in with '{:.5f}'
            + dict arrowprops:
                usage: pass in `arrowprops=dict(arrowstyle='->', color='r')`
    + `color_palette(n, cmap="jet")`: return a series of color, color_list can be seen by typing print(mgt2001.notes.color_list.__doc__)
    + `add_margin(ax, x=0.05, y=0.05)`: set the margin for ax
    + `Class Pos()`:
        Gotta init with 
        `posObj = Pos(len(df['Education']), len(year_label), space=0.3)`

        You can get:
        + `posObj.width`
        + `posObj.returnPos()`: will return the positions as a list
        + `posObj.restructure(len(df['Education']), len(df['Year']))` ```can restructure the data

    ----------------

    ## `per.py`
    per == percentage-related stuff

    + `percentile(data, percentage)`: will return the value corresponding to the percentile from the data.
    + `percentrank(data, value)`: will return the corresponding percentile rank given the value.

    ----------------

    ## `des.py`
    des == describe or description

    + `interval(df)`: given a dataframe or an array, return the interval for the data. Mainly for Empirical Rules and Chebyshev.
    + `outlier(data, base=per)`: given a df col as the data; base is the function is should use. 
        For numpy, please enter `np`. Put in nothing for the `per` using the per.percentile() as defined in the textbook.
    + `kurtosis(df)`: return the kurtosis of the df col
    + `convert_excel_kurtosis(K, n)`: will return the real kurtosis; n being the number of samples
    + `skew(df)`: return the skewness of the df col
    + `convert_excel_skew(G, n)`: will return the real skewness; n being the number of samples

    ----------------

    ## `prob.py`

    + `BayesTHM(pre_probs, event='D')`: 
        Output: (Bayes):
        >>> [num1, num2] >> meaning that given event B has occurred, the probability that the previous event occurred is num1
        ============
        Input:
        >>> BayesTHM(np.array[[0.9, 0.1], # being the previous conditions
                                  [0.99, 0.9], # being the conditional probabilities: ([P(B | A), P(B | A̅)])
                                  [0.01, 0.1]]) # being the complementaries of the conditional probabilities: ([P(B̅ | A), P(B̅ | A̅)])

        The deafult number of columns is 2.

        Pass in `event='B'` to specify the post event you are to compare.
    + `portfolio_analysis(stock_df, target_stocks, pss)`:
        Return portfolios DataFrame as a list

        Usage:
        df = pd.read_excel('Xr07-TSE.xlsx')
        df = df.set_index(['Year', 'Month'])
        month_dict = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
        df.rename(index=month_dict, inplace=True)

        target_stocks = ['BMO', 'MG', 'POW', 'RCL.B']
        pss = np.array([[.25, .25, .25, .25], [.2,.6, .1, .1], [.1, .2, .3, .4]])

        portfolios = portfolio_analysis(df[target_stocks], target_stocks, pss)

        >> notes for np.cov and np.corrcoef
        cov_mat = np.cov(stock_df[target_stocks].values, rowvar=False) # [i, i] = variance of each stock
        cor_mat = np.corrcoef(df[target_stocks].values, rowvar = False)
    + `covariance(x, mu_x, y, mu_y, prob)`: 
        Usage: `covariance(ibm_x, ibm.expect(), ms_y, ms.expect(), prob)`

        Where `prob` equals:

        !!! notes
            → x  
            ↓ y

        Prob can be a two dimensional array or a `numpy` array.


    ----------------

    ## `notes.py`
    Here are a list of notes:
    + measure_of_movement
    + kurtosis
    + skewness
    + color_list
    + README <-- you are looking at this one!

    They can be regarded as a cheat sheet. Use wisely.
    """
