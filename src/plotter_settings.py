#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 11:44:47 2023

@author: jaya
"""

import sys
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def set_plot_settings(pres_type):
    if pres_type == 'talk':
        s = 16
    elif pres_type == 'paper':
        s = 10
    mpl.rc('font',family='sans-serif')
    mpl.rc('font',size=s)
    mpl.rc('font',size=s)
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['axes.formatter.useoffset'] = False
    mpl.rcParams['ytick.right'] = False
    mpl.rcParams['xtick.top'] = False
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['legend.fontsize'] = s
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['axes.labelsize'] = s
    mpl.rcParams['xtick.labelsize'] = s
    mpl.rcParams['ytick.labelsize'] = s
    mpl.rcParams['xtick.major.pad']=  2 #3.5
    mpl.rcParams['ytick.major.pad']=  2 #3.5
    mpl.rcParams['axes.labelpad'] = 1 #4.0
    mpl.rcParams['legend.handlelength'] = 1.0#2.0
    mpl.rcParams['legend.handletextpad'] = 0.4# 0.8
    mpl.rcParams['legend.columnspacing'] = 1.2# 2.0,
    mpl.rcParams['lines.markersize'] = 4.0
def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)