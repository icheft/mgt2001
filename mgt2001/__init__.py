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


def color_palette(n, cmap="jet"):
    '''
    Generate a series of color using matplotlib color map templates.
    The default color map (cmap) is set to "jet".
    '''

    get_color = getattr(cm, cmap)
    return get_color(np.linspace(0,1,n))


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
            self.pos.append([j - (1 - self.space) / 2. + i * self.width for j in range(1,self.groupedN+1)])

    def returnPos(self):
        return self.pos

    def restructure(self, groupedN, seriesN):
        self.groupedN = groupedN
        self.seriesN = seriesN
        self.width = (1 - self.space) / (self.groupedN)
        self.__calcPos()