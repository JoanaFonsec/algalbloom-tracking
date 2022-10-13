#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

# Python imports
from array import array
import time
import math
import numpy as np
import scipy.io
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist
from scipy.spatial import distance
import signal

# Utils
from utils import Utils

class LSQEstimator:
    def __init__(self, s, range_m,, earth_radius=6369345):

        self.__kernel = gp.kernels.ConstantKernel(44.29588721)*gp.kernels.Matern(length_scale=[0.54654887, 0.26656638])
        self.s = s
        self.__model = gp.GaussianProcessRegressor(kernel=self.__kernel, optimizer=None, alpha=self.s**2)

        # Estimation range where to predict values
        self.range_deg = range_m / (np.radians(1.0) * earth_radius)

    """
    Least Squares Regression - Gradient analytical estimation

    Parameters
    ----------
    X:self.trajectory coordinates array
    y: self.measurements on X coordinates
    dist_metric: distance metric used to calculate distances
    """
    def est_grad(self, X, y, dist_metric='euclidean'):
        self.__model.fit(X[:-1], y[:-1])
        x = np.atleast_2d(X[-1])

        dists = cdist(x/length_scale, X[:-1]/length_scale, metric=dist_metric)

        dists = dists * np.sqrt(3)

        x_dist = Utils.nonabs_1D_dist(x[:,0], X[:-1,0]) / (length_scale[0]**2)
        y_dist = Utils.nonabs_1D_dist(x[:,1], X[:-1,1]) / (length_scale[1]**2)

        common_term = -3 * sigma * np.exp(-dists)

        dx = x_dist * common_term
        dy = y_dist * common_term

        return np.matmul(dx,self.__model.alpha_) , np.matmul(dy,self.__model.alpha_) 