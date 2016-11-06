#!/usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn.apionly as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve, GridSearchCV

from icecube import ShowerLLH

# Plotting decision regions


def plot_decision_regions(X, y, classifier, resolution=0.02, scatter_fraction=0.025):
    # setup marker generator and color map
    markers = ('s', '^', 'o', '^', 'v')
    colors = ('b', 'g', 'r', 'y', 'c')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    if scatter_fraction != None:
        fraction_event_selection_mask = (np.random.uniform(0, 1, len(y)) <= scatter_fraction)
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[(y == cl) & fraction_event_selection_mask, 0],
                        y=X[(y == cl) & fraction_event_selection_mask, 1],
                        alpha=0.5, c=cmap(idx),
                        marker=markers[idx], label=cl)

def histogram_2D(x, y, bins, log_counts=False, make_prob=False, ax=None, **opts):
    h, xedges, yedges = np.histogram2d(x, y, bins=bins, normed=False)
    h = np.rot90(h)
    h = np.flipud(h)
    h = np.ma.masked_where(h == 0, h)
    if make_prob:
        ntot = np.sum(h, axis=0).astype('float')
        ntot[ntot == 0] = 1.
        h /= ntot
    if log_counts:
        h = np.log10(h)
    extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
    colormap = 'viridis'
    if ax != None:
        im = ax.imshow(h, extent=extent, origin='lower',
                   interpolation='none', cmap=colormap)
    else:
        im = plt.imshow(h, extent=extent, origin='lower',
                   interpolation='none', cmap=colormap)
    # if log_counts:
    #     plt.colorbar(im, label='$\log_{10}(\mathrm{Counts})$')
    # else:
    #     plt.colorbar(im, label='$Counts$')
    if not make_prob and not log_counts:
        cb = plt.colorbar(im, label='$Counts$')
    if not make_prob and log_counts:
        cb = plt.colorbar(im, label='$\log_{10}(\mathrm{Counts})$')

    return im
