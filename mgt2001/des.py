from scipy.stats import moment
import math as math
import mgt2001.per as per


def outlier(data):
    Q1 = per.percentile(data, 25)
    Q2 = per.percentile(data, 50)
    Q3 = per.percentile(data, 75)
    IQR = Q3 - Q1  # IQR is interquartile range.
    print("Q1 = ", Q1)
    print("Q2 = ", Q2)
    print("Q3 = ", Q3)
    print("IQR = ", IQR)
    filter = (data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)
    if (filter.size != 0):
        outlier_prompt = "Outliers are listed as follows:\n{}".format(
            data.loc[filter])
    else:
        outlier_prompt = "There are no outliers."

    description = """Q1 = {}
    Q2 = {}
    Q3 = {}
    IQR = {}

    {}
    """.format(Q1, Q2, Q3, IQR, outlier_prompt)

    return description


def kurtosis(df):
    """
    How to read the kurtosis value?

    K > 3 : Leptokurtic (Narrow-tall)
    K = 3 : Mesokurtic (Regular)
    K < 3 : Platykurtic (Wide-low)
    -------
    To convert excel kurtosis to a real one, please consider the function "convert_excel_kurtosis(K, n)"

    """
    m2 = moment(df, moment=2)
    m4 = moment(df, moment=4)
    kurtosis_f = m4 / pow(m2, 2)
    return kurtosis_f


def convert_excel_kurtosis(K, n):
    """
    K is the kurtosis you get from Excel.

    The function will return (n-2) * (n-3) * K / ((n + 1) * (n - 1)) + 3 * (n - 1) / (n + 1).
    """
    if (n < 3):
        return None
    return (n-2) * (n-3) * K / ((n + 1) * (n - 1)) + 3 * (n - 1) / (n + 1)


def skew(df):
    """
    How to read skew value?

    g > 0 : Skewed to right
    g = 0 : Symmetric
    g < 0 : Skewed to left
    -------
    To convert excel skewness to a real one, please consider the function "convert_excel_skew(G, n)"
    """
    m2 = moment(df, moment=2)
    m3 = moment(df, moment=3)
    skew_f = m3 / pow(pow(m2, 0.5), 3)
    return skew_f


def convert_excel_skew(G, n):
    """
    G is the skewness you get from Excel.

    The function will return (n-2) * G / math.sqrt(n * (n - 1)).
    """
    if (n < 2):
        return None
    return (n-2) * G / math.sqrt(n * (n - 1))
