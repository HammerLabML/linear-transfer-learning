"""
Implements linear supervised transfer learning according to the
paper Paaßen et al. (2018). Expectation maximization transfer learning
and its application for bionic hand prostheses. Neurocomputing, 298, 122-133.
doi:10.1016/j.neucom.2017.11.072. (available at
https://arxiv.org/abs/1711.09256).

Copyright (C) 2019
Benjamin Paaßen
AG Machine Learning
Bielefeld University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from sklearn.base import BaseEstimator

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '0.1.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class SLGMM_transfer_model(BaseEstimator):
    """ A linear transfer learning model based on a labelled Gaussian
    mixture model with shared precision matrix in the source space.

    Attributes:
    model:      A labelled Gaussian mixture model with shared precision
                matrix (slGMM) for the source space.
    regul:      The L2 regularization strength. 1E-5 per default.
    error_delta: A stopping criterion based on the error. If the error
                improves by less than this, the algorithm terminates.
                1E-5 per default.
    max_it:     The maximum number of iterations for the expectation
                maximization scheme. 50 per default.
    _H:         The transfer matrix H, mapping from the target to the
                souce space
    _loss:      The error curve during the last training run. Each entry is the
                negative log likelihood after an expectation step.
    """
    def __init__(self, model, regul = 1E-5, error_delta = 1E-5, max_it = 50):
        self.model = model
        self.regul = regul
        self.error_delta = error_delta
        self.max_it = max_it


    def fit(self, X, y):
        """ Trains a transfer mapping H such that the given data X, y
        fits to the slGMM of this model after applying H.

        Args:
        X: A n_samples, n_features matrix of data
        y: A n_samples vector of labels.

        Returns: This instance.
        """
        N = X.shape[0]
        n = X.shape[1]
        m = self.model._Mus.shape[0]
        K = self.model._Mus.shape[1]
        L = len(self.model._labels)

        # initialize H as identity
        self._H = np.eye(m, n)

        # already prepare the pseudo-inverse of the data, which we can
        # re-use during training
        X_pinv = np.dot(X, np.linalg.inv(np.dot(X.T, X) + self.regul * np.eye(n)))

        # for each data point, compute the a priori probability to be generated
        # by each component, just based on priors and class probabilities
        Prior = np.zeros((K, N))
        for l in range(L):
            Prior[:, y == self.model._labels[l]] = np.expand_dims(self.model._P_Y[l, :] * self.model._Pi, 1)

        # now, start the actual EM algorithm
        self._loss = []
        for i in range(self.max_it):
            # First, the expectation step.

            # compute the squared distances between Gaussian means
            # and data
            Dsq = np.full((K, N), np.inf)
            # iterate over all components
            for k in range(K):
                # compute the squared distances between mean and data
                # for all relevant points
                relevant_k = Prior[k, :] > 1E-5
                Delta = np.dot(X[relevant_k, :], self._H.T) - np.expand_dims(self.model._Mus[:, k], axis=0)
                Dsq[k, relevant_k] = np.sum(Delta * np.dot(Delta, self.model._Lambda), axis=1)

            # compute the Gamma matrix from the squared distances
            # iterate over all components
            Gamma = np.zeros((K, N))
            for k in range(K):
                # compute the non-normalized gamma values
                relevant_k = Prior[k, :] > 1E-5
                # if no point is relevant, continue
                if(not np.any(relevant_k)):
                    continue
                # get the minimum for this component for numeric reasons
                dsq_min = np.min(Dsq[k, relevant_k])
                # perform the computation
                Gamma[k, relevant_k] = np.exp(-0.5 * (Dsq[k, relevant_k] - dsq_min)) * Prior[k, relevant_k]
            # normalize to obtain the posterior p(k|x, y)
            Gamma /= np.expand_dims(np.sum(Gamma, axis=0), axis=0)

            # now, compute the weighted squared error
            valid  = np.logical_and(Gamma > 1E-5, np.logical_not(np.isnan(Gamma)))
            loss = 0.5 * np.sum(Gamma[valid] * Dsq[valid])

            # check if the loss decreased enough to continue
            if(self._loss and self._loss[-1] - loss < self.error_delta):
                # otherwise, return directly
                return self
            self._loss.append(loss)

            # now, perform the maximization step
            self._H = np.dot(self.model._Mus, np.dot(Gamma, X_pinv))

        # after all iterations are over, return
        return self

    def predict(self, X):
        """ Applies the learned linear transformation to the input data.

        Args:
        X: A n_samples, n_features matrix of data

        Returns: X * H.T
        """
        return np.dot(X, self._H.T)
