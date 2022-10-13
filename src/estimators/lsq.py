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

        lsq_solution = np.linalg.lstsq(X,y)
        
        return lsq_solution