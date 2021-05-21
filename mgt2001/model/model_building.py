def fst_order_wt_int(x, a, b, c):
    return a + b * x[0] + c * x[1]


def fst_order_w_int(x, a, b, c, d):
    return a + b * x[0] + c * x[1] + d * x[0] * x[1]


def snd_order_wt_int(x, a, b, c, d, e):
    return a + b * x[0] + c * x[1] + d * x[0] ** 2 + e * x[1] ** 2


def snd_order_w_int(x, a, b, c, d, e, f):
    return a + b * x[0] + c * x[1] + d * x[0] ** 2 + e * x[1] ** 2 + f * x[0] * x[1]
