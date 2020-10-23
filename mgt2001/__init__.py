from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import math as math
from adjustText import adjust_text

import mgt2001.per as per
import mgt2001.des as des
import mgt2001.notes as notes
import mgt2001.tree as tree
import mgt2001.prob as prob

from mgt2001._version import __version__


def geomean(rate):
    """
    Result: Will return the geometric mean

    Usage:
    return_rate = np.array([0.2, 0.1, -0.05])
    avg_rate = geomean(return_rate)
    print("geo_mean =", avg_rate)
    """
    rate_1 = rate + 1
    geo_m = math.exp(np.log(rate_1).mean()) - 1
    return geo_m


def ori_autolabel(rects, truncate='{}'):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate(truncate.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')


def autolabel(rects, **kwargs):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    list of kwargs:
        + bool original
        + string truncate: in the '{}' format. Can be filled in with '{:.5f}'
        + dict arrowprops:
            usage: pass in `arrowprops=dict(arrowstyle='->', color='r')`
    """
    if "original" in kwargs and kwargs["original"] == True:
        if "truncate" in kwargs:
            ori_autolabel(rects, kwargs["truncate"])
            return
        else:
            ori_autolabel(rects)
        return

    texts = []
    for rect in rects:
        height = rect.get_height()
        if "truncate" in kwargs:
            texts.append(plt.text(s=kwargs["truncate"].format(height),
                                  x=(rect.get_x() + rect.get_width() / 2),
                                  y=(height)))
        else:
            texts.append(plt.text(s='{}'.format(height),
                                  x=(rect.get_x() + rect.get_width() / 2),
                                  y=(height)))

    if "arrowprops" in kwargs:
        adjust_text(texts, arrowprops=kwargs['arrowprops'])


def color_palette(n, cmap="jet"):
    '''
    Generate a series of color using matplotlib color map templates.
    The default color map (cmap) is set to "jet".
    '''

    get_color = getattr(cm, cmap)
    return get_color(np.linspace(0, 1, n))


def add_margin(ax, x=0.05, y=0.05):
    '''
     Setting margins in matplotlib/seaborn with subplots.

     This will, by default, add 5% to the x and y margins. You
     can customize this using the x and y arguments when you call it.
     ref: https://stackoverflow.com/a/34205235/10871988

     e.g.
     ax = sns.regplot(x='Temperature', y= 'Tickets', data = df, ci = None, scatter_kws={'color':'dodgerblue'}, line_kws={'linestyle':'dashed', 'color':'#ffaa77'})
     add_margin(ax, x=0.01, y=0.00) ### Call this after regplot 
    '''

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xmargin = (xlim[1]-xlim[0])*x
    ymargin = (ylim[1]-ylim[0])*y

    ax.set_xlim(xlim[0]-xmargin, xlim[1]+xmargin)
    ax.set_ylim(ylim[0]-ymargin, ylim[1]+ymargin)


class Pos():
    # positioning - src: http://emptypipes.org/2013/11/09/matplotlib-multicategory-barchart/
    def __init__(self, groupedN, seriesN, space=0.3):
        self.space = space
        self.groupedN = groupedN
        self.seriesN = seriesN
        self.width = (1 - self.space) / (self.groupedN)
        self.__calcPos()

    def __calcPos(self):
        self.pos = list()
        for i in range(self.seriesN):
            self.pos.append([j - (1 - self.space) / 2. + i *
                             self.width for j in range(1, self.groupedN+1)])

    def returnPos(self):
        return self.pos

    def restructure(self, groupedN, seriesN):
        self.groupedN = groupedN
        self.seriesN = seriesN
        self.width = (1 - self.space) / (self.groupedN)
        self.__calcPos()
