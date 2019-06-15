import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt

def simplePlot(x, y, xLabel='X', yLabel='Y', plotName='default', symbol='.',
               errorX=None, errorY=None, figsize=(10,10)):
    if errorX is None:
        xerr = 0
    else:
        xerr=errorX
    if errorY is None:
        yerr = 0
    else:
        yerr=errorY
    fig, ax = plt.subplots(1,1,figsize=figsize)
    ax.plot(x, y, symbol)
    ax.errorbar(x, y, yerr=yerr, xerr=xerr, linestyle='')
    ax.grid()
    ax.set_ylabel(yLabel, fontsize=20)
    ax.set_xlabel(xLabel, fontsize=20)
    ax.tick_params(direction='in', length=5, width=1, colors='k')

    fig.tight_layout()
    fig.savefig(plotName + '.png', dpi = 300)
    return fig, ax
