from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from adjustText import adjust_text

def ori_autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def autolabel(rects, **kwargs):
    """
    Attach a text label above each bar in *rects*, displaying its height.
    Only *arrowprops* is available right now.
    """
    if "original" in kwargs and kwargs["original"] == True:
        ori_autolabel(rects)
        return

    texts = []
    for rect in rects:
        height = rect.get_height()
        texts.append(plt.text(s='{}'.format(height),
                    x=(rect.get_x() + rect.get_width() / 2),
                    y=(height)))

    if "arrowprops" in kwargs:
        adjust_text(texts, arrowprops=kwargs['arrowprops'])
    else: 
        adjust_text(texts)


def color_palette(n, cmap="jet"):
    '''
    Generate a series of color using matplotlib color map templates.
    The default color map (cmap) is set to "jet".
    '''

    get_color = getattr(cm, cmap)
    return get_color(np.linspace(0,1,n))