from numpy.testing._private.utils import measure
import pandas as pd
import numpy as np


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
