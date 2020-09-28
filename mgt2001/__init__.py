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
    texts = []
    for rect in rects:
        height = rect.get_height()
        texts.append(plt.text(s='{}'.format(height),
                    x=(rect.get_x() + rect.get_width() / 2),
                    y=(height)))
    
    if len(kwargs):
        adjust_text(texts, arrowprops=kwargs['arrowprops'])
    else: 
        adjust_text(texts)
