__author__ = 'Oleksandra'

from sklearn.utils import array2d, as_float_array, atleast2d_or_csr

from scipy import linalg

import numpy as np


class SPCA():

    def __init__(self):

        self.variance_percent_retained = 0.99

        # min number of principal components to maintain self.variance_percent_retained
        self.k_components = 1

        # array of components with maximum variance
        self.components = None

        self.mean_ = None

        self.U = None
        self.S = None
        self.V = None

        self.U_reduce = None
    #--------------------------------------------------------------------------------------------------------------

    def transform(self, features):

        features = atleast2d_or_csr(features)

        if self.mean_ is not None:
            features = features - self.mean_

        features = np.dot(features, self.U_reduce);

        ##features = safe_sparse_dot(features, self.components.T)
        ## features = np.dot(np.transpose(self.U[:, :self.k_components]), features)

        return features
    #--------------------------------------------------------------------------------------------------------------

    def fit(self, features_train):

        X = array2d(features_train)
        n_samples, n_features = X.shape
        print 'given train features dimensions before PCA : ', features_train.shape

        X = as_float_array(X)

        # Data preprocessing by Mean Normalization
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        # Feature scaling
        std_dev = np.std(X, axis=0)
        X = np.divide(X, std_dev)

        # Compute covariance matrix
        cov_matrix = np.dot(np.transpose(X), X)/n_samples
        print 'cov_matrix dimensions : ', cov_matrix.shape
        # Compute SVD
        U, S, V = linalg.svd(cov_matrix, full_matrices=1, compute_uv=1)
        print 'x dimensions : ', X.shape
        print 'U dimensions : ', U.shape
        print 'S dimensions : ', S.shape

        # Calculate optimal k - min number of principal components to maintain 99% of variance
        variance_retained = np.sum(S[:self.k_components]) / np.sum(S)

        while variance_retained < self.variance_percent_retained:
            self.k_components += 1
            variance_retained = np.sum(S[:self.k_components]) / np.sum(S)
            #print 'k_components : ', self.k_components, ' variance : ', variance_retained

        if self.k_components is None:
            self.k_components = n_features
        elif not 0 <= self.k_components <= n_features:
            raise ValueError("n_components=%r invalid for n_features=%d" % (self.k_components, n_features))

        self.components = U

        self.U_reduce = U[:, :self.k_components]
        print 'number of principal components : ', self.k_components

        self.U = U
        self.S = S
        self.V = V

        return (U, S, V)
    #--------------------------------------------------------------------------------------------------------------

