from matplotlib import pyplot as plt
import itertools
import statsmodels.formula.api as smf
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.cm as cm
import statsmodels.api as sm


def _add_margin(ax, x=0.05, y=0.05):

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xmargin = (xlim[1]-xlim[0])*x
    ymargin = (ylim[1]-ylim[0])*y

    ax.set_xlim(xlim[0]-xmargin, xlim[1]+xmargin)
    ax.set_ylim(ylim[0]-ymargin, ylim[1]+ymargin)


def fst_order_wt_int(x, a, b, c):
    return a + b * x[0] + c * x[1]


def fst_order_w_int(x, a, b, c, d):
    return a + b * x[0] + c * x[1] + d * x[0] * x[1]


def snd_order_wt_int(x, a, b, c, d, e):
    return a + b * x[0] + c * x[1] + d * x[0] ** 2 + e * x[1] ** 2


def snd_order_w_int(x, a, b, c, d, e, f):
    return a + b * x[0] + c * x[1] + d * x[0] ** 2 + e * x[1] ** 2 + f * x[0] * x[1]


def _color_palette(n, cmap="jet"):
    '''
    Generate a series of color using matplotlib color map templates.
    The default color map (cmap) is set to "jet".
    '''

    get_color = getattr(cm, cmap)
    return get_color(np.linspace(0, 1, n))


def multi_variable_plot(x_name=None, y_name=None, x_names: list = None, indicator=None, df=None, cmap='tab20b', label: dict = None, fit=False, df_result=None, **kwargs):
    """
    e.g.
    fit = False:
        label={1: 'White', 2: 'Silver', 3: 'Other Colors'}
        x_name = 'Odometer'
        y_name = 'Price'
        indicator = 'Color'
    fit = True:
        label={1: 'White', 2: 'Silver', 3: 'Other Colors'},
        df=df_onehot
        x_name = 'Odometer',
        y_name = 'Price',
        indicator = 'Color',
        x_names = ['Odometer', 'color_1', 'color_2'],  # 1
        df_result=res_dict['df_result'], # df_result = df_result
        x = [{'b0': [0, 2], 'b1': [1, 4]},
            {'b0': [0, 3], 'b1': [1, 5]},
            {'b0': [0], 'b1': [1]}] # 手動輸入哪些項目是要被加入的（go by params (start with constant))
        # ind_start = 1 (start with first independent variable)
    """
    if x_names is not None:
        n = len(x_names)
        color = _color_palette(n, cmap)
        fig, ax = plt.subplots()
        for i, (col_name) in enumerate(x_names):
            _ = sns.regplot(
                x=x_name, y=y_name, data=df[df[col_name] == 1], color=color[i], ci=None, label=col_name)

        plt.legend()
        plt.title(f'Scatter Plot for {x_name} and {y_name}')
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        _add_margin(ax, x=0.02, y=0.00)  # Call this after tsplot
        plt.show()
    else:
        n = len(label)
        color = _color_palette(n, cmap)
        if fit == False:
            fig, ax = plt.subplots()
            for i, (key, value) in enumerate(label.items()):
                _ = sns.regplot(
                    x=x_name, y=y_name, data=df[df[indicator] == key], color=color[i], ci=None, label=value)

            plt.legend()
            plt.title(f'Scatter Plot for {x_name} and {y_name}')
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            _add_margin(ax, x=0.02, y=0.00)  # Call this after tsplot
            plt.show()
        else:
            fig, ax = plt.subplots()

            x_ind = df_result.params.index
            X_plot = np.linspace(df[x_name].min(),
                                 df[x_name].max(), 100)
            k = len(label) - 1  # 2

            try:
                ind_start = kwargs['ind_start']
                ind_end = ind_start + len(label) - 2
            except:
                ind_start = -1

            try:
                snd_ind_start = kwargs['snd_ind_start']
                snd_ind_end = snd_ind_start + len(label) - 2
            except:
                snd_ind_start = -1

            try:
                x_iter = kwargs['x']
                x_iter_flag = True
            except:
                x_iter_flag = False

            for i, (key, value) in enumerate(label.items()):
                _ = sns.scatterplot(
                    x=x_name, y=y_name, data=df[df[indicator] == key], color=color[i], label=value)

                if ind_start != -1:
                    if ind_start <= ind_end:
                        b0 = df_result.params[0] + \
                            df_result.params[ind_start + 1]
                        ind_start += 1
                    else:
                        b0 = df_result.params[0]
                else:
                    b0 = df_result.params[0]

                if snd_ind_start != -1:
                    if snd_ind_start <= snd_ind_end:
                        b1 = df_result.params[1] + \
                            df_result.params[snd_ind_start + 1]
                        snd_ind_start += 1
                    else:
                        b1 = df_result.params[1]
                else:
                    b1 = df_result.params[1]

                x_iter_str = ""
                if x_iter_flag:
                    # [{'b0': (0, 2), 'b1': (1, 4)}, {'b0': (0, 3), 'b1': (1, 5)}, {'b0': (0), 'b1': (1)}]
                    b0 = b1 = 0
                    for b0_param in x_iter[i]['b0']:
                        b0 += df_result.params[b0_param]
                    for b1_param in x_iter[i]['b1']:
                        b1 += df_result.params[b1_param]
                    x_iter_str += f'\n$b_0$: ({", ".join(list(map(str, x_iter[i]["b0"])))}) / $b_1$: ({", ".join(list(map(str, x_iter[i]["b1"])))})'

                Y_plot = b0 + b1 * X_plot
                if x_iter_flag:
                    plt.plot(X_plot, Y_plot, color=color[i],
                             label=f'$\hat y = {b0:.2f} + {b1:2f}x$ ({x_iter_str})')
                else:
                    plt.plot(X_plot, Y_plot, color=color[i],
                             label=f'$\hat y = {b0:.2f} + {b1:2f}x$')

            # plt.legend()
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            plt.title('Fit Lines')
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            plt.show()


def LogisticRegression(x_names=None, y_name=None, df=None, alpha=0.05, precision=4, show_summary=True):
    res_dict = dict()
    y_data = df[y_name]
    X_data_T = np.array(df[x_names])
    X_data = pd.DataFrame(X_data_T, columns=x_names)
    X_data_update = sm.add_constant(X_data)

    logit_model = sm.Logit(y_data, X_data_update)
    df_result = logit_model.fit()
    res_dict['df_result'] = df_result
    if show_summary:
        print(df_result.summary())
        print()

    return res_dict


def LogisticRegressionPrediction(candidates=None, x1=None, df_result=None, precision=4):
    """
    x1 is passed when you only want to go one by one
    candidates can store a dictionary like {'gmat': [590,740,680,610,710],
                  'gpa': [2,3.7,3.3,2.3,3],
                  'work_experience': [3,4,6,1,5]}
    """
    if x1 is not None:
        x1.insert(0, 1)
        # print(x1)
        y_pred = df_result.predict(x1)
        S_hat = np.exp(y_pred)
        P_A = S_hat / (S_hat + 1)
        Result_ar = np.array([y_pred, S_hat, P_A])
        Result_D = pd.DataFrame(Result_ar.T, columns=[
                                'y_hat', 'S_hat', 'Probability'])
        display_str = f'''======= Logistic Regression Prediction =======
When
'''
        for i in range(len(x1) - 1):
            display_str += f'  + {df_result.params.index[i + 1]} = {x1[i + 1]}\n'
        display_str += f'''
Probability = {Result_D['Probability'][0]:.{precision}f}
'''
        print(display_str)
        df = Result_D
    else:
        df = pd.DataFrame(candidates, columns=candidates.keys())
        df = sm.add_constant(df)

        y_pred = df_result.predict(df)
        S_hat = np.exp(y_pred)
        P_A = S_hat / (S_hat + 1)
        Result_ar = np.array([y_pred, S_hat, P_A])
        Result_D = pd.DataFrame(Result_ar.T, columns=[
                                'y_hat', 'S_hat', 'Probability'])
        df = pd.concat([df, Result_D], axis=1)
        display(df)

    return df


def stepwise_selection(df, y_name="y", x_names=["x1"], verbose=False):
    """
    using ols (str) instead of OLS (array)
    """
    selected = []
    candidates = x_names.copy()
    best_adjr2 = -1
    best_subset = []
    cnt = 0
    while len(candidates) > 0:
        cnt += 1
        if verbose:
            print("Current Candidates: ", candidates)
        tmp_indep_subset = []
        tmp_model_adjr = []
        tmp_model_nindep = []
        for acandidate in candidates:
            tmplist = selected.copy()
            tmplist.append(acandidate)
            modelstr = y_name + " ~ " + "+".join(tmplist)
            result6tmp = smf.ols(modelstr, data=df).fit()
            tmp_indep_subset.append(tmplist)
            tmp_model_adjr.append(result6tmp.rsquared_adj)
            tmp_model_nindep.append(len(tmplist))
        tmp_adjr2 = np.array(tmp_model_adjr)
        tmpind = tmp_adjr2.argmax()
        this_adjr2 = tmp_adjr2[tmpind]
        selected = tmp_indep_subset[tmpind]
        if this_adjr2 <= 0:
            raise("Encounterd negative Adj R2. Stop.")
        if verbose:
            print("===============")
            print("Current best model: ", selected)
            print("Current best AdjR2: ", this_adjr2)
        if this_adjr2 > best_adjr2:
            #print(" best result updated")
            best_adjr2 = this_adjr2
            best_subset = selected
        candidates = set(candidates) - set(selected)
        candidates = list(candidates)
    print(f'''======= Stepwise Regression Selection =======
Stop after {cnt} iterations.
''')
    print("Best adjR2 = ", best_adjr2)
    print("Best subset = ", best_subset)
    print()

    res_dict = dict()
    res_dict['best_subset'] = best_subset
    res_dict['best_adjR2'] = best_adjr2
    modelstr = y_name + " ~ " + "+".join(best_subset)
    result6b = smf.ols(modelstr, data=df).fit()
    res_dict['best_model'] = result6b
    print(result6b.summary())
    return res_dict


def best_subset_selection(df, y_name='y', x_names=['x1'], verbose=False, visualize=False):
    def processSubset(y_v, X_v, feature_set):
        X_v_a = sm.add_constant(X_v[list(feature_set)])
        model = sm.OLS(y_v, X_v_a)
        regr = model.fit()
        RSS = regr.rsquared_adj
        if verbose:
            print("Current Candidates: ", list(regr.params.index[1:]))
            # print("===============")
            print("Current AdjR2: ", regr.rsquared_adj)
        return {"model": regr, "RSS": RSS}

    def getBest(y_g, X_g, k):
        iter_cnt = 0
        results = []
        for combo in itertools.combinations(X_g.columns, k):
            iter_cnt += 1
            results.append(processSubset(y_g, X_g, combo))
        models = pd.DataFrame(results)
        best_model = models.loc[models['RSS'].argmax()]
        return best_model, iter_cnt

    y_var = df[y_name]
    X_var = df[x_names]
    cnt = 0
    models_best = pd.DataFrame(columns=["RSS", "model"])
    for i in range(1, len(x_names) + 1):
        models_best.loc[i], new_cnt = getBest(y_var, X_var, i)
        cnt += new_cnt

    Fb = models_best[models_best['RSS'] == models_best.RSS.max()].index.values
    best_model = models_best.loc[Fb[0], "model"]
    best_subset, best_adjr2 = list(
        best_model.params.index[1:]), best_model.rsquared_adj
    print(f'''======= Best Subset Regression Selection =======
Stop after {cnt} iterations.
''')
    print("Best AdjR2 = ", best_adjr2)
    print("Best subset = ", best_subset)
    print()
    res_dict = dict()
    res_dict['best_subset'] = best_subset
    res_dict['best_adjR2'] = best_adjr2
    res_dict['all_model_df'] = models_best
    print(best_model.summary())
    res_dict['best_model'] = best_model
    if visualize:
        rsquared_adj = models_best.apply(
            lambda row: row[1].rsquared_adj, axis=1)
        fig, ax = plt.subplots()
        plt.plot(rsquared_adj)
        plt.plot(rsquared_adj.max())
        for a in Fb:
            plt.axvline(x=a, color='purple', linestyle='--')
        plt.xlabel('# Predictors')
        plt.ylabel('adjusted rsquared')
        plt.show()
    return res_dict
