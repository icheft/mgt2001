import scipy.stats as stats


def f_test(std1, n1, std2, n2, siglevel):
    fstat = (std1 * n1 / (n1 - 1))**2 / (std2 * n2 / (n2 - 1))**2
    fcv = stats.f.ppf(1-siglevel/2, n1 - 1, n2 - 1)
    return fstat, fcv


def t_test(m1, std1, n1, m2, std2, n2, c, siglevel):
    sp2 = ((n1 - 1)*(std1 * n1 / (n1 - 1))**2 + (n2 - 1)
           * (std2 * n2 / (n2 - 1))**2) / (n1 + n2 - 2)
#     tstat = (m1 - m2 - c)/((std1**2/n1 + std2**2/n2))**0.5
    tstat = (m1 - m2 - c)/(sp2*(1/n1 + 1/n2))**0.5
    tcv = stats.t.ppf(1-siglevel/2, df=n1 + n2 - 2)
    return tstat, tcv


def t_test_onetail(m1, std1, n1, m2, std2, n2, c, siglevel):
    sp2 = ((n1 - 1)*(std1 * n1 / (n1 - 1))**2 + (n2 - 1)
           * (std2 * n2 / (n2 - 1))**2) / (n1 + n2 - 2)
#     tstat = (m1 - m2 - c)/((std1**2/n1 + std2**2/n2))**0.5
    tstat = (m1 - m2 - c)/(sp2*(1/n1 + 1/n2))**0.5
    tcv = stats.t.ppf(1-siglevel, df=n1 + n2 - 2)
    return tstat, tcv
