#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

# Python imports
import numpy as np
from scipy.spatial.distance import cdist

# Utils
from utils import Utils

class LSQEstimator:
    def __init__(self, s, range_m, earth_radius=6369345):

        self.s = s
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

        if len(X[0]) != len(X[1]):
            raise ValueError("Estimation: Coordinates dimensions does not match.")

        if len(X[0]) != len(y):
            raise ValueError("Estimation: y must have the same length as x.")

        # A = [x 1]
        A = np.vstack([X[0], X[1], np.ones(len(X[0]))]).T

        # y = [delta_vector, delta_zero]
        y = y[:, np.newaxis]

        # LSQ takes in positions and measurements and returns the gradient
        self.alpha, self.beta, self.delta_zero = np.linalg.lstsq(A, y, rcond=None)[0].squeeze()

        return self.alpha, self.beta, self.delta_zero