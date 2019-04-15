import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt

def simplePlot(x, y, errorX, errorY, xLabel, yLabel, plotName):
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    ax.plot(x, y, '.')
    ax.errorbar(x, y, yerr = errorY, xerr = errorX, linestyle="None")
    ax.grid()
    ax.set_ylabel(yLabel, fontsize=20)
    ax.set_xlabel(xLabel, fontsize=20)
    ax.tick_params(direction='in', length=5, width=1, colors='k')
    
    fig.tight_layout()
    fig.savefig(plotName + '.png', dpi = 300)
    return fig, ax
