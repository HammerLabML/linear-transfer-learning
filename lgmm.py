"""
Implements labelled Gaussian mixture models in a scikit-learn
compatible fashion.

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

import re
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '0.1.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class SLGMM(BaseEstimator, ClassifierMixin):
    """ A labelled Gaussian mixture model with shared precision matrices.

    Note that this is _not_ the same as a standard Gaussian mixture model,
    because labelled Gaussian mixture models support a label distribution
    for each component (also refer to the _P_Y) parameter. As such, this
    model is also _not_ equivalent to training a separate GMM for each label
    because this model can cope with 'fuzzy' assignments of components to
    multiple labels.

    Attributes:
    K:          The number of Gaussian components. It is also possible to
                input a string her to specify a number of Gaussian components
                per class of the form 'K per class'. '1 per class' per default.
    error_delta: A stopping criterion based on the error. If the error
                improves by less than this, the algorithm terminates.
                1E-5 per default.
    max_it:     The maximum number of iterations for the expectation
                maximization scheme. 50 per default.
    min_sigma:  The minimum standard deviation in each dimension to prevent
                a degeneration of _Lambda. 1E-3 per default.
    _dim:       The dimensionality m of the vector space. Initialized according
                to the data.
    _Mus:       A _dim x K matrix containing the means for all Gaussian
                components as columns. Initialized randomly in the data span.
    _Lambda:    A _dim x _dim matrix containing the shared precision matrix.
                Initialized as the identity.
    _Pi:        A K-dimensional vector containing the prior for all components.
                Initialized as 1 / K for all components.
    _labels:    An L-dimensional vector containing labels for each label index.
    _P_Y:       A L x K matrix containing the class-distribution for each
                component. Initialized either as strict assignment or uniformly,
                depending on the value for K.
    _loss:      The error curve during the last training run. Each entry is the
                negative log likelihood after an expectation step.
    """
    def __init__(self, K = '1 per class', error_delta = 1E-5, max_it = 50, min_sigma = 1E-3):
        self.K = K
        self.error_delta = error_delta
        self.max_it = max_it
        self.min_sigma = min_sigma

    def fit(self, X, y):
        """ Trains a labelled Gaussian mixture model for the given input
        data.

        Args:
        X: A n_samples, n_features matrix of data
        y: A n_samples vector of labels.

        Returns: This instance.
        """
        M = X.shape[0]
        # set the dimensionality
        self._dim = X.shape[1]
        # retrieve the unique labels
        self._labels = np.sort(np.unique(y))
        L = len(self._labels)
        # translate label vector into standard format
        y2 = np.zeros((M), dtype=int)
        for l in range(L):
            y2[y == self._labels[l]] = l

        # check the setting for K and initialize the Gaussian components
        # accordingly
        if(isinstance(self.K, str)):
            # if K is a string, check if it is formatted correctly
            match = re.match('(\d+) per class', self.K)
            if(match is None):
                raise ValueError('K was not formatted correctly. Expected the form "K per class" but got %s.' % self.K)
            K_per_class = int(match.group(1))
            # initialize the class distributions crisply
            self._P_Y = np.zeros((L, K_per_class * L))
            for l in range(L):
                self._P_Y[l, l * K_per_class:(l+1)*K_per_class] = 1.
            K = K_per_class * L
        elif(isinstance(self.K, int)):
            # initialize the class distributions uniformly
            self._P_Y = np.ones((L, self.K)) / float(L)
            K = self.K
        else:
            raise ValueError('K had unknown type; expected int or string')

        # initialize the means in the convex hull of the data, but take
        # class distributions into account
        Gamma = np.random.rand(K, M)
        for l in range(L):
            Gamma[:, y2 == l] *= np.expand_dims(self._P_Y[l, :], 1)
        # normalize row-wise
        Gamma /= np.expand_dims(np.sum(Gamma, axis=1), 1)
        # the means result as product from Gamma and X
        self._Mus = np.dot(Gamma, X).T

        # initialize the precision matrix as the identity
        self._Lambda = np.eye(self._dim)

        # initialize the prior uniformly
        self._Pi = np.ones(K) / K

        # now, start the actual EM algorithm
        self._loss = []
        for i in range(self.max_it):
            # First, the expectation step.

            # as preparation, compute which data point is a-priori
            # relevant for which Gaussian component due to class distributions
            R = np.full((K, M), False, dtype=bool)
            for k in range(K):
                for l in np.where(self._P_Y[:, k] > 1E-5)[0]:
                    R[k, y2 == l] = True

            # compute the squared distances between Gaussian means
            # and data
            Dsq = np.full((K, M), np.inf)
            # iterate over all components
            for k in range(K):
                # compute the squared distances between mean and data
                # for all relevant points
                Delta = X[R[k, :], :] - np.expand_dims(self._Mus[:, k], axis=0)
                Dsq[k, R[k, :]] = np.sum(Delta * np.dot(Delta, self._Lambda), axis=1)

            # compute the Gamma matrix from the squared distances
            # iterate over all components
            Gamma = np.zeros((K, M))
            for k in range(K):
                # get the minimum for this component for numeric reasons
                dsq_min = np.min(Dsq[k, R[k, :]])
                # compute the non-normalized gamma values
                ks  = np.exp(-0.5 * (Dsq[k, R[k, :]] - dsq_min))
                pys = self._P_Y[y2[R[k, :]], k]
                Gamma[k, R[k, :]] =  ks * pys * self._Pi[k]
            # normalize to obtain the posterior p(k|x, y)
            Gamma /= np.expand_dims(np.sum(Gamma, axis=0), axis=0)

            # now, compute the negative log likelihood sum_i -log(p(x_i, y_i)),
            # which is given as the free energy (which we minimize in the
            # maximization step) minus the entropy of gamma for all i
            valid  = np.logical_and(Gamma > 1E-5, np.logical_not(np.isnan(Gamma)))
            sqloss = 0.5 * np.sum(Gamma[valid] * Dsq[valid])
            lambdaloss = - 0.5 * M * np.log(np.linalg.det(self._Lambda)) + 0.5 * M * self._dim * np.log(2 * np.pi)
            piloss = - np.sum(np.dot(np.log(self._Pi), Gamma))
            pyloss = 0.
            for k in range(K):
                pyloss -= np.dot(Gamma[k, R[k, :]], np.log(self._P_Y[y2[R[k, :]], k]))
            gamma_nnzs = Gamma[Gamma > 0]
            entropy = - np.dot(gamma_nnzs, np.log(gamma_nnzs))
            # add all the parts to achieve the final loss
            loss = sqloss + lambdaloss + piloss + pyloss + entropy

            # check if the loss decreased enough to continue
            if(self._loss and self._loss[-1] - loss < self.error_delta):
                # otherwise, return directly
                return self
            self._loss.append(loss)

            # now, perform the maximization step
            # update priors
            self._Pi = np.mean(Gamma, axis=1)
            # update class distributions
            for l in range(L):
                self._P_Y[l, :] = np.sum(Gamma[:, y2 == l], axis=1) / np.sum(Gamma, axis=1)
            # update means
            self._Mus = (np.dot(Gamma, X) / np.expand_dims(np.sum(Gamma, axis=1), 1)).T
            # compute the new covariance matrix
            Sigma = np.zeros((self._dim, self._dim))
            for k in range(K):
                # compute the difference between mean and data for all
                # relevant points
                Delta = X[R[k, :], :] - np.expand_dims(self._Mus[:, k], axis=0)
                # compute the covariance for the current component, weighted
                # by Gamma
                Sigma += np.dot((Delta * np.expand_dims(Gamma[k, R[k, :]], axis=1)).T, Delta)
            # normalize Sigma by M
            Sigma /= M
            # perform an eigenvalue decomposition of Sigma to ensure that
            # the covariance in no direction degenerates
            Eigs, V = np.linalg.eig(Sigma)
            # repair any numeric complex values
            V = np.real(V)
            Eigs = np.real(Eigs)
            # bound the eigenvalues from below
            Eigs[Eigs < self.min_sigma ** 2] = self.min_sigma ** 2
            # then construct the precision matrix via inversion
            self._Lambda = np.dot(V, np.dot(np.diag(1. / Eigs), V.T))
        # after all iterations are over, return
        return self


    def predict_proba(self, X):
        """ Predicts class probabilities for all input data.

        Args:
        X: A n_samples, n_features matrix of data.

        Returns: a n_samples x n_classes matrix of class probabilities.
        """
        M = X.shape[0]
        K = self._Mus.shape[1]
        # compute the squared distances between Gaussian means
        # and data
        Dsq = np.full((K, M), np.inf)
        # iterate over all components
        for k in range(K):
            Delta = X - np.expand_dims(self._Mus[:, k], axis=0)
            Dsq[k, :] = np.sum(Delta * np.dot(Delta, self._Lambda), axis=1)

        # subtract the minimum distance from all distances
        Dsq_normalized = Dsq - np.expand_dims(np.min(Dsq, axis=1), axis=1)

        # compute the assignment factors from all Gaussian components to
        # all data points; these are essentially the probabilities
        # p(x, k) for all x and k, ignoring the label
        Gamma = np.exp(-0.5 * Dsq_normalized) * np.expand_dims(self._Pi, 1)

        # By multiplying with P(y|k) we obtain p(x, y) for all x and y
        P = np.dot(self._P_Y, Gamma)
        # In a last step, we just need to transpose and normalize over the
        # classes to obtain the desired class probabilities
        P = P.T / np.expand_dims(np.sum(P.T, axis=1), 1)
        # return
        return P


    def predict(self, X):
        """ Predicts class labels for all input data.

        Args:
        X: A n_samples, n_features matrix of data.

        Returns: a n_samples vector of predicted class labels
        """
        P = self.predict_proba(X)
        # get the most likely label index for each point
        ls = np.argmax(P, axis=1)
        # return the label for each index
        return self._labels[ls]

def slgmm_from_lvq(lvq_model, sigma = 1.):
    """ Converts a (GM)LVQ model to a compatible SLGMM.

    The input LVQ model is converted to an SLGMM by generating one Gaussian
    component per prototype with mean = prototype position and a crisp
    class distribution with probability one for the prototype's class and
    zero everywhere else. The metric learning matrix omega is converted to
    a precision matrix via omega.T * omega.

    Args:
    lvq_model:  The input LVQ model. We assume that this model has the
                attributes w_ containing the prototypes, c_w_ containing
                the prototype labels, and, optionally, omega_ for the
                projection matrix and lambda_ for relevances.
                Please refer to the documentation at
                https://mrnuggelz.github.io/sklearn-lvq/
                for more information.
    sigma:      A scaling parameter for the precision matrix. A smaller
                sigma means that the classification behavior of the converted
                SLGMM becomes more crisp and thus more similar to the original
                LVQ behavior. For transfer learning purposes, this parameter
                is irrelevant, though, and is, thus, set to 1 per default.
    """
    # set up an 'empty' slgmm
    model = SLGMM(K = lvq_model.w_.shape[0])
    # set up the Gaussian means
    model._dim = lvq_model.w_.shape[1]
    model._Mus = lvq_model.w_.T
    # set up the Lambda matrix
    if(not hasattr(lvq_model, 'omega_')):
        if(not hasattr(lvq_model, 'lambda_')):
            model._Lambda = np.eye(model._dim)
        else:
            model._Lambda = np.diag(lvq_model.lambda_ ** 2)
    else:
        if(not hasattr(lvq_model, 'lambda_')):
            model._Lambda = np.dot(lvq_model.omega_.T, lvq_model.omega_)
        else:
            omega_lambda  = np.dot(lvq_model.omega_, np.diag(lvq_model.lambda_))
            model._Lambda = np.dot(omega_lambda.T, omega_lambda)
    # scale lambda with sigma
    model._Lambda /= sigma ** 2
    # set up priors
    model._Pi = np.ones(model.K) / model.K
    # set up class distributions
    model._labels = np.sort(np.unique(lvq_model.c_w_))
    L = len(model._labels)
    model._P_Y = np.zeros((L, model.K))
    for l in range(L):
        model._P_Y[l, lvq_model.c_w_ == model._labels[l]] = 1.
    # return the model
    return model


class LGMM(BaseEstimator, ClassifierMixin):
    """ A labelled Gaussian mixture model.

    Attributes:
    K:          The number of Gaussian components. It is also possible to
                input a string her to specify a number of Gaussian components
                per class of the form 'K per class'. '1 per class' per default.
    error_delta: A stopping criterion based on the error. If the error
                improves by less than this, the algorithm terminates.
                1E-5 per default.
    max_it:     The maximum number of iterations for the expectation
                maximization scheme. 50 per default.
    min_sigma:  The minimum standard deviation in each dimension to prevent
                a degeneration of _Lambda. 1E-3 per default.
    _dim:       The dimensionality m of the vector space. Initialized according
                to the data.
    _Mus:       A _dim x K matrix containing the means for all Gaussian
                components as columns. Initialized randomly in the data span.
    _Lambdas:   A _dim x _dim x K tensor containing the precision matrices.
                Initialized as the identity for all components.
    _sqrtdets:  A K-dimensional vector containing the square-rooted determinants
                for all precision matrices. These are normalized by the smallest
                determinant to prevent numeric degeneration.
    _Pi:        A K-dimensional vector containing the prior for all components.
                Initialized as 1 / K for all components.
    _labels:    An L-dimensional vector containing labels for each label index.
    _P_Y:       A L x K matrix containing the class-distribution for each
                component. Initialized either as strict assignment or uniformly,
                depending on the value for K.
    _loss:      The error curve during the last training run. Each entry is the
                negative log likelihood after an expectation step.
    """
    def __init__(self, K = '1 per class', error_delta = 1E-5, max_it = 50, min_sigma = 1E-3):
        self.K = K
        self.error_delta = error_delta
        self.max_it = max_it
        self.min_sigma = min_sigma

    def fit(self, X, y):
        """ Trains a labelled Gaussian mixture model for the given input
        data.

        Args:
        X: A n_samples, n_features matrix of data
        y: A n_samples vector of labels.

        Returns: This instance.
        """
        M = X.shape[0]
        # set the dimensionality
        self._dim = X.shape[1]
        # retrieve the unique labels
        self._labels = np.sort(np.unique(y))
        L = len(self._labels)
        # translate label vector into standard format
        y2 = np.zeros((M), dtype=int)
        for l in range(L):
            y2[y == self._labels[l]] = l

        # check the setting for K and initialize the Gaussian components
        # accordingly
        if(isinstance(self.K, str)):
            # if K is a string, check if it is formatted correctly
            match = re.match('(\d+) per class', self.K)
            if(match is None):
                raise ValueError('K was not formatted correctly. Expected the form "K per class" but got %s.' % self.K)
            K_per_class = int(match.group(1))
            # initialize the class distributions crisply
            self._P_Y = np.zeros((L, K_per_class * L))
            for l in range(L):
                self._P_Y[l, l * K_per_class:(l+1)*K_per_class] = 1.
            K = K_per_class * L
        elif(isinstance(self.K, int)):
            # initialize the class distributions uniformly
            self._P_Y = np.ones((L, self.K)) / float(L)
            K = self.K
        else:
            raise ValueError('K had unknown type; expected int or string')

        # initialize the means in the convex hull of the data, but take
        # class distributions into account
        Gamma = np.random.rand(K, M)
        for l in range(L):
            Gamma[:, y2 == l] *= np.expand_dims(self._P_Y[l, :], 1)
        # normalize row-wise
        Gamma /= np.expand_dims(np.sum(Gamma, axis=1), 1)
        # the means result as product from Gamma and X
        self._Mus = np.dot(Gamma, X).T

        # initialize the precision matrices as the identity
        self._Lambdas = np.zeros((self._dim, self._dim, K))
        for k in range(K):
            self._Lambdas[:, :, k] = np.eye(self._dim)
        # as a helper construct, we also need the determinant of each
        # precision matrix, which is one initially
        self._sqrtdets = np.ones(K)

        # initialize the prior uniformly
        self._Pi = np.ones(K) / K

        # now, start the actual EM algorithm
        self._loss = []
        for i in range(self.max_it):
            # First, the expectation step.

            # as preparation, compute which data point is a-priori
            # relevant for which Gaussian component due to class distributions
            R = np.full((K, M), False, dtype=bool)
            for k in range(K):
                for l in np.where(self._P_Y[:, k] > 1E-5)[0]:
                    R[k, y2 == l] = True

            # compute the squared distances between Gaussian means
            # and data
            Dsq = np.full((K, M), np.inf)
            # iterate over all components
            for k in range(K):
                # compute the squared distances between mean and data
                # for all relevant points
                Delta = X[R[k, :], :] - np.expand_dims(self._Mus[:, k], axis=0)
                Dsq[k, R[k, :]] = np.sum(Delta * np.dot(Delta, self._Lambdas[:, :, k]), axis=1)

            # compute the Gamma matrix from the squared distances
            # iterate over all components
            Gamma = np.zeros((K, M))
            min_sqrtdet = np.min(self._sqrtdets)
            for k in range(K):
                # get the minimum for this component for numeric reasons
                dsq_min = np.min(Dsq[k, R[k, :]])
                # compute the non-normalized gamma values
                ks  = (self._sqrtdets[k] / min_sqrtdet) * np.exp(-0.5 * (Dsq[k, R[k, :]] - dsq_min)) 
                pys = self._P_Y[y2[R[k, :]], k]
                Gamma[k, R[k, :]] =  ks * pys * self._Pi[k]
            # normalize to obtain the posterior p(k|x, y)
            Gamma /= np.expand_dims(np.sum(Gamma, axis=0), axis=0)

            # now, compute the negative log likelihood sum_i -log(p(x_i, y_i)),
            # which is given as the free energy (which we minimize in the
            # maximization step) minus the entropy of gamma for all i
            valid  = np.logical_and(Gamma > 1E-5, np.logical_not(np.isnan(Gamma)))
            sqloss = 0.5 * np.sum(Gamma[valid] * Dsq[valid])
            lambdaloss = - np.sum( np.dot( Gamma.T, np.log(self._sqrtdets))) + 0.5 * M * self._dim * np.log(2 * np.pi)
            piloss = - np.sum(np.dot(np.log(self._Pi), Gamma))
            pyloss = 0.
            for k in range(K):
                pyloss -= np.dot(Gamma[k, R[k, :]], np.log(self._P_Y[y2[R[k, :]], k]))
            gamma_nnzs = Gamma[Gamma > 0]
            entropy = - np.dot(gamma_nnzs, np.log(gamma_nnzs))
            # add all the parts to achieve the final loss
            loss = sqloss + lambdaloss + piloss + pyloss + entropy

            # check if the loss decreased enough to continue
            if(self._loss and self._loss[-1] - loss < self.error_delta):
                # otherwise, return directly
                return self
            self._loss.append(loss)

            # now, perform the maximization step
            # update priors
            self._Pi = np.mean(Gamma, axis=1)
            # update class distributions
            for l in range(L):
                self._P_Y[l, :] = np.sum(Gamma[:, y2 == l], axis=1) / np.sum(Gamma, axis=1)
            # update means
            self._Mus = (np.dot(Gamma, X) / np.expand_dims(np.sum(Gamma, axis=1), 1)).T
            # compute the new precision matrices
            for k in range(K):
                # compute the difference between mean and data for all
                # relevant points
                Delta = X[R[k, :], :] - np.expand_dims(self._Mus[:, k], axis=0)
                # compute the covariance for the current component, weighted
                # by Gamma
                Sigma = np.dot((Delta * np.expand_dims(Gamma[k, R[k, :]], axis=1)).T, Delta) / np.sum(Gamma[k, R[k, :]])
                # perform an eigenvalue decomposition of Sigma to ensure that
                # the covariance in no direction degenerates
                Eigs, V = np.linalg.eig(Sigma)
                # repair any numeric complex values
                V = np.real(V)
                Eigs = np.real(Eigs)
                # bound the eigenvalues from below
                Eigs[Eigs < self.min_sigma ** 2] = self.min_sigma ** 2
                # then construct the precision matrix via inversion
                self._Lambdas[:, :, k] = np.dot(V, np.dot(np.diag(1. / Eigs), V.T))
                # and recompute the square-rooted determinants
                self._sqrtdets[k] = np.sqrt(np.prod(1. / Eigs))
        # after all iterations are over, return
        return self


    def predict_proba(self, X):
        """ Predicts class probabilities for all input data.

        Args:
        X: A n_samples, n_features matrix of data.

        Returns: a n_samples x n_classes matrix of class probabilities.
        """
        M = X.shape[0]
        K = self._Mus.shape[1]
        # compute the squared distances between Gaussian means
        # and data
        Dsq = np.full((K, M), np.inf)
        # iterate over all components
        for k in range(K):
            Delta = X - np.expand_dims(self._Mus[:, k], axis=0)
            Dsq[k, :] = np.sum(Delta * np.dot(Delta, self._Lambdas[:, :, k]), axis=1)

        # subtract the minimum distance from all distances
        Dsq_normalized = Dsq - np.expand_dims(np.min(Dsq, axis=1), axis=1)

        # compute the assignment factors from all Gaussian components to
        # all data points; these are essentially the probabilities
        # p(x, k) for all x and k, ignoring the label
        Gamma = np.exp(-0.5 * Dsq_normalized) * np.expand_dims(self._Pi, 1) * np.expand_dims(self._sqrtdets / np.min(self._sqrtdets), 1)

        # By multiplying with P(y|k) we obtain p(x, y) for all x and y
        P = np.dot(self._P_Y, Gamma)
        # In a last step, we just need to transpose and normalize over the
        # classes to obtain the desired class probabilities
        P = P.T / np.expand_dims(np.sum(P.T, axis=1), 1)
        # return
        return P


    def predict(self, X):
        """ Predicts class labels for all input data.

        Args:
        X: A n_samples, n_features matrix of data.

        Returns: a n_samples vector of predicted class labels
        """
        P = self.predict_proba(X)
        # get the most likely label index for each point
        ls = np.argmax(P, axis=1)
        # return the label for each index
        return self._labels[ls]


def lgmm_from_lvq(lvq_model, sigma = 1., min_rel = 1E-3):
    """ Converts an LGMLVQ model to a compatible LGMM.

    The input LVQ model is converted to an LGMM by generating one Gaussian
    component per prototype with mean = prototype position and a crisp
    class distribution with probability one for the prototype's class and
    zero everywhere else. The metric learning matrices omega are converted to
    precision matrices via omega.T * omega.

    Args:
    lvq_model:  The input LVQ model. We assume that this model has the
                attributes w_ containing the prototypes, c_w_ containing
                the prototype labels, and omegas_ for the
                projection matrices.
                Please refer to the documentation at
                https://mrnuggelz.github.io/sklearn-lvq/
                for more information.
    sigma:      A scaling parameter for the precision matrix. A smaller
                sigma means that the classification behavior of the converted
                LGMM becomes more crisp and thus more similar to the original
                LVQ behavior. For transfer learning purposes, this parameter
                is irrelevant, though, and is, thus, set to 1 per default.
    min_rel:    The minimum relevance in each dimension to prevent a
                degeneration of the determinant of a precision matrix.
                1E-3 per default.
    """
    # set up an 'empty' lgmm
    model = LGMM(K = lvq_model.w_.shape[0])
    # set up the Gaussian means
    model._dim = lvq_model.w_.shape[1]
    model._Mus = lvq_model.w_.T
    # set up the Lambda matrices
    model._Lambdas = np.zeros((model._dim, model._dim, model.K))
    model._sqrtdets = np.zeros(model.K)
    for k in range(model.K):
        model._Lambdas[:, :, k] = np.dot(lvq_model.omegas_[k].T, lvq_model.omegas_[k]) / sigma ** 2
        # to compute the determinant, we first retrieve the eigenvalues
        eigs = np.real(np.linalg.eigvals(model._Lambdas[:, :, k]))
        eigs[eigs < min_rel] = min_rel
        model._sqrtdets[k] = np.sqrt(np.prod(eigs))
    # set up priors
    model._Pi = np.ones(model.K) / model.K
    # set up class distributions
    model._labels = np.sort(np.unique(lvq_model.c_w_))
    L = len(model._labels)
    model._P_Y = np.zeros((L, model.K))
    for l in range(L):
        model._P_Y[l, lvq_model.c_w_ == model._labels[l]] = 1.
    # return the model
    return model
