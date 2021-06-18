from numpy.testing._private.utils import measure
import statsmodels.stats.outliers_influence as sso
import statsmodels.api as sm
import pandas as pd
import numpy as np


def cma(y_v, period: int):
    y_v_MA_a = np.zeros(len(y_v))
    y_v_MA_a[:] = np.nan
    n = period
    if n % 2 == 0:  # even
        halfwin = int(n / 2)
        y_v_MA_ta = np.zeros(len(y_v))
        # first attempt
        for i in range(halfwin, len(y_v) - halfwin + 1):
            y_v_MA_ta[i] = np.mean(y_v[(i-halfwin): (i+halfwin)])
        # second attempt
        for ii in range(halfwin, len(y_v) - halfwin):
            y_v_MA_a[ii] = np.mean(y_v_MA_ta[(ii): (ii+2)])
    else:  # odd
        halfwin = int((n - 1) / 2)
        for i in range(halfwin, len(y_v) - halfwin):
            y_v_MA_a[i] = np.mean(y_v[(i-halfwin): (i+halfwin+1)])
    return y_v_MA_a


def seasonal_index(y_v, period: int, show=False, option='cma'):
    """
    return seasonal index and a Series with all observations linked to its seasonal index
    """
    n = period
    if option == 'cma':
        SI_MA_a = np.zeros(len(y_v))
        SI_MA_a[:] = np.nan
        SI_MA_a = y_v / cma(y_v, period)
        SI_id_s = np.arange(1, len(y_v)+1)
        SI_id = SI_id_s - np.floor(SI_id_s / n) * n
        SI_id[np.where((SI_id[:] == 0))] = n
        SI_MA_df = pd.DataFrame({'SIMA': SI_MA_a, 'SIid': SI_id})
        SI_MA_u = np.zeros(n)
        for j in range(1, n+1):
            SI_MA_u[j-1] = SI_MA_df['SIMA'][SI_MA_df['SIid']
                                            == j].dropna().mean()
        SI_MA = SI_MA_u / sum(SI_MA_u) * n
        if show:
            print('Seasonal Index:', SI_MA)
        return SI_MA, SI_MA_df['SIid']
    elif option == 'lr':
        y_data = y_v
        X_data_ar = np.arange(1, len(y_v)+1)
        X_data_T = X_data_ar.T
        X_data = pd.DataFrame(X_data_T, columns=['Time'])
        X_data = sm.add_constant(X_data)
        olsmod = sm.OLS(y_data, X_data)
        result_reg = olsmod.fit()
        st, data, ss2 = sso.summary_table(result_reg, alpha=0.05)
        y_v_LR_a = data[:, 2]
        SI_LR_a = y_v / y_v_LR_a
        SI_id_s = np.arange(1, len(y_v)+1)
        SI_id = SI_id_s - np.floor(SI_id_s / n) * n
        SI_id[np.where((SI_id[:] == 0))] = n
        SI_LR_a_df = pd.DataFrame({'SILR': SI_LR_a, 'SIid': SI_id})
        SI_LR_u = np.zeros(n)
        for j in range(1, n+1):
            SI_LR_u[j-1] = SI_LR_a_df['SILR'][SI_LR_a_df['SIid']
                                              == j].dropna().mean()
        SI_LR = SI_LR_u / sum(SI_LR_u) * n

        if show:
            print('Seasonal Index:', SI_LR)

        return SI_LR, SI_LR_a_df['SIid']


def deseasonalize(y_v, period: int, show=False, option='cma'):
    y_v_SI_MA = np.zeros(len(y_v))
    DSI_y_v = np.zeros(len(y_v))
    if option == 'cma':
        SI_MA, SIid = seasonal_index(y_v, period, show, option=option)

        for k in range(0, len(y_v)):
            Idd = int(SIid[k] - 1)
            y_v_SI_MA[k] = SI_MA[Idd]
            DSI_y_v[k] = y_v[k] / SI_MA[Idd]
        if show:
            print('Deseasonalized Data:', DSI_y_v)

        SI_MA_result_m = np.array([SIid, y_v_SI_MA, y_v, DSI_y_v])
        SI_MA_result_df = pd.DataFrame(SI_MA_result_m.T, columns=[
            'SID', 'SeaIdx', 'orig', 'Des_Y'])

        return {'deasonalized_y': DSI_y_v, 'seasonal_i': SIid, 'des_result': SI_MA_result_df, 'SI_MA': SI_MA}
    elif option == 'lr':
        y_v_SI_LR = np.zeros(len(y_v))
        DSI_y_v = np.zeros(len(y_v))
        SI_LR, SIid = seasonal_index(y_v, period, show, option=option)
        for k in range(0, len(y_v)):
            Idd = int(SIid[k] - 1)
            y_v_SI_LR[k] = SI_LR[Idd]
            DSI_y_v[k] = y_v[k] / SI_LR[Idd]

        if show:
            print('Deseasonalized Data:', DSI_y_v)
        SI_LR_result_a = np.array([SIid, y_v_SI_LR, y_v, DSI_y_v])
        SI_LR_result_df = pd.DataFrame(SI_LR_result_a.T, columns=[
                                       'SID', 'SeaIdx', 'orig', 'Des_Y'])
        return {'deasonalized_y': DSI_y_v, 'seasonal_i': SIid, "des_result": SI_LR_result_df, "SI_LR": SI_LR}


def smoothing_cma(df, y_name, time_name, period, show=False):
    DSI_y_v, SIid, des_result, SI_MA = deseasonalize(
        df[y_name], period, show, option='cma').values()
    des_result.rename(columns={'Des_Y': f'Des_{y_name}'}, inplace=True)
    return pd.concat([df, des_result], axis=1)


def smoothing_lr(df, y_name, time_name, period, show=False):
    DSI_y_v, SIid, des_result, SI_LR = deseasonalize(
        df[y_name], period, show, option='lr').values()
    des_result.rename(columns={'Des_Y': f'Des_{y_name}'}, inplace=True)
    return pd.concat([df, des_result], axis=1)


def smoothing(df, y_name, time_name, period, show=False, option='cma'):
    DSI_y_v, SIid, des_result, SI = deseasonalize(
        df[y_name], period, show, option=option).values()
    des_result.rename(columns={'Des_Y': f'Des_{y_name}'}, inplace=True)
    return pd.concat([df, des_result], axis=1)


def seasonal_prediction(df, df_result, y_name, time_name, new_t, period, show=False, option='cma'):
    """
    df should be the deseasoned df if possible
    """
    y_v = df[y_name]
    if len(new_t) == 0:
        _, data, _ = sso.summary_table(df_result, alpha=0.05)
        trend_proj = df_result.predict(sm.add_constant(new_t))
        df_result.predict(sm.add_constant(new_t))
        tdf = df.copy()
        tdf[f'Pre_{y_name}'] = data[:, 2] * tdf['SeaIdx']
        return tdf
    else:
        new_t = np.array(new_t)
        SI, SIid = seasonal_index(y_v, period, show, option=option)
        # des_df = smoothing_cma(df, y_name, time_name,
        #                        period=period, show=show)  # final df secured

        trend_proj = df_result.predict(sm.add_constant(new_t))
        seasonal_adj = trend_proj * SI
        # new_t = np.arange(12, 16)
        _, data, _ = sso.summary_table(df_result, alpha=0.05)
        trend_proj = df_result.predict(sm.add_constant(new_t))
        df_result.predict(sm.add_constant(new_t))
        tdf = df.copy()
        tdf[f'Pre_{y_name}'] = data[:, 2] * tdf['SeaIdx']

        # tdf[x_name] = np.append(tdf[x_name], new_t)
        for i, t in enumerate(new_t):
            tdf = tdf.append({time_name: t, 'SID': tdf['SID'].values[-(1 + i) - (len(new_t) - (1 + i))], 'SeaIdx': tdf['SeaIdx'].values[-(1 + i) - (
                len(new_t) - (1 + i))], f'Pre_{y_name}': trend_proj[i] * tdf['SeaIdx'].values[-(1 + i) - (len(new_t) - (1 + i))]}, ignore_index=True)

    return tdf


def mav(df, n, y_name, time_name, weight: list = None):
    """
    mgt2001.ts.mav
    """
    ma_p_a = np.zeros(df.shape[0] + 1)
    y_v = df[y_name].dropna().values
    ma_p_a[:] = np.nan
    for i in range(n, len(ma_p_a)):
        if weight is not None:
            ma_p_a[i] = np.sum(y_v[i-n: i] * np.array(weight))
        else:
            ma_p_a[i] = np.mean(y_v[i-n: i])
    t1 = np.append(df[time_name].dropna().values[:len(y_v)], [
                   df[time_name].dropna().values[:len(y_v)][-1] + 1])

    ma_df_p = pd.DataFrame(
        {time_name: t1, y_name: np.append(y_v, [np.nan]), f'MA({n})': ma_p_a})
    return pd.concat([ma_df_p, df.drop(columns=[time_name, y_name])], axis=1)


def exp_smoothing(df, alpha, y_name, time_name):
    y_v = df[y_name].dropna().values
    es_df = pd.DataFrame({'orig': y_v})
    es_df['es_res'] = es_df['orig'].ewm(alpha=alpha, adjust=False).mean()
    es_al_a = np.zeros(len(y_v) + 1)
    es_al_a[0] = es_df['es_res'][0]
    es_al_a[1] = es_df['es_res'][0]
    for i in range(2, len(y_v) + 1):
        es_al_a[i] = es_df['es_res'][i-1]
    t1 = np.append(df[time_name].dropna().values[:len(y_v)], [
                   df[time_name].dropna().values[:len(y_v)][-1] + 1])
    org_data = y_v.tolist()
    es_df_e = pd.DataFrame(
        {time_name: t1, y_name: np.append(y_v, [np.nan]), f'ES({alpha})': es_al_a})
    return pd.concat([es_df_e, df.drop(columns=[time_name, y_name])], axis=1)


def exp_trend(df, alpha, beta, average, trend, n=2, y_name="", time_name="", drop=True):
    """
>>> esm_df = mgt2001.ts.exp_trend(esm_df, 0.3, 0.3, 38, 1, n=2, y_name='Enrollments', time_name='Time')
>>> esm_df = esm_df.merge(mgt2001.ts.exp_trend(df, 0.3, 0.5, 38, 1, n=2, y_name='Enrollments', time_name='Time'))

    Set drop to False, if a full trend DataFrame is needed
    """
    from statsmodels.tsa.api import Holt
    import statsmodels.api as sm

    y_v = df[y_name].dropna().values
    esm_a = np.array(y_v)
    esm_model = Holt(esm_a, initialization_method='known', initial_level=average, initial_trend=trend).fit(
        smoothing_level=alpha, smoothing_trend=beta, optimized=False)
    esm_fit = esm_model.fittedvalues
    esm_fcast = esm_model.forecast(n)
    esm_ab_a = np.zeros(len(y_v) + n + 1)
    esm_ab_a[0] = average + trend
    for i in range(1, len(y_v) + 1):
        esm_ab_a[i] = esm_fit[i-1]
    for i in range(len(y_v) + 1, len(y_v) + n + 1):
        esm_ab_a[i] = esm_fcast[i-len(y_v)-1]

    Ini_v = average + trend
    if drop:
        t1 = np.append(df[time_name].dropna().values[:len(y_v)], [
            df[time_name].dropna().values[:len(y_v)][-1] + i for i in range(1, n + 1)])
        org_data = np.append(y_v, [np.nan]*n)
        wsm_df_ab = pd.DataFrame(
            {time_name: t1, f'{y_name}': org_data, f'EST({alpha}, {beta}, {average}, {trend})': esm_ab_a[1:]})
        old_df = df.drop(columns=[time_name, y_name])
        # na_df = pd.DataFrame(
        #     [[np.nan for i in range(len(old_df.columns))]], columns=old_df.columns)
        # old_df = pd.concat([na_df, old_df], ignore_index=True)
        for i in range(n):
            na_df = pd.DataFrame(
                [[np.nan for i in range(len(old_df.columns))]], columns=old_df.columns)
            old_df = pd.concat([old_df, na_df], ignore_index=True)
    else:
        t1 = np.append(df[time_name].dropna().values[:len(y_v)], [
            df[time_name].dropna().values[:len(y_v)][-1] + i for i in range(1, n + 2)])
        W_ini_v = np.append(Ini_v, y_v)
        org_data = np.append(W_ini_v, [np.nan]*n)
        wsm_df_ab = pd.DataFrame(
            {time_name: t1, f'{y_name}': org_data, f'EST({alpha}, {beta}, {average}, {trend})': esm_ab_a[:]})
        old_df = df.drop(columns=[time_name, y_name])
        na_df = pd.DataFrame(
            [[np.nan for i in range(len(old_df.columns))]], columns=old_df.columns)
        old_df = pd.concat([na_df, old_df], ignore_index=True)
        for i in range(n):
            na_df = pd.DataFrame(
                [[np.nan for i in range(len(old_df.columns))]], columns=old_df.columns)
            old_df = pd.concat([old_df, na_df], ignore_index=True)

    return pd.concat([wsm_df_ab, old_df], axis=1)


def error_metric(y=None, pred_y=None, df: pd.DataFrame = None, measurements: list = None, y_name: str = '', precision=4):
    def compute_error(y, pred_y):
        raw_err = np.array(y - pred_y)
        err = raw_err[~np.isnan(raw_err)]
        raw_percentage = np.array((y - pred_y) / y)
        err_percentage = raw_percentage[~np.isnan(raw_percentage)]

        MAD = np.absolute(err).mean()
        MSE = np.mean(err * err)
        RMSE = np.sqrt(np.mean(err * err))
        MAPE = np.absolute(err_percentage).mean() * 100

        err_dict = {'MAD': MAD, 'MSE': MSE, 'RMSE': RMSE, 'MAPE': MAPE}

        return err_dict

    if measurements is None:

        err_dict = compute_error(y, pred_y)

        result = f"""======= Error Metrics =======
MAD = {err_dict['MAD']:.{precision}f}
MSE = {err_dict['MSE']:.{precision}f}
RMSE = {err_dict['RMSE']:.{precision}f}
MAPE = {err_dict['MAPE']:.{precision}f}%
"""
        print(result)
        return err_dict
    else:
        err_df = pd.DataFrame(columns=measurements, index=[
                              'MAD', 'MSE', 'RMSE', 'MAPE'])
        df = df.dropna()
        for measurement in measurements:
            y = df[y_name]
            pred_y = df[measurement]
            err_dict = compute_error(y, pred_y)

            result = f"""### Error Metrics - {measurement}
MAD = {err_dict['MAD']:.{precision}f}
MSE = {err_dict['MSE']:.{precision}f}
RMSE = {err_dict['RMSE']:.{precision}f}
MAPE = {err_dict['MAPE']:.{precision}f}%
"""
            print(result)

            err_df[measurement] = list(err_dict.values())

        min_val = err_df.min(axis=1).values

        def _highlight(val):
            color = 'lightgreen' if val in min_val else 'default'
            return 'background-color: %s' % color

        style_df = err_df.style.applymap(_highlight)

        return err_df, style_df
