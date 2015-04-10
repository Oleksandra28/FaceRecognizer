__author__ = 'Oleksandra'


from sklearn.datasets import fetch_olivetti_faces

from sklearn.cross_validation import  train_test_split
from sklearn.metrics import accuracy_score

from sklearn.utils import as_float_array
from sklearn import linear_model

import matplotlib.pyplot as plt

import numpy as np

from SPCA import SPCA
from SLR import SLR

from visualization import *



def classification_log_reg(features_train_, features_test_, labels_train_, labels_test_, labels_, dataset_, thetas_):

    dataset = dataset_
    labels = labels_

    n_samples, h, w = dataset.images.shape
    faces_images = dataset.images

    features_train = features_train_
    features_test = features_test_
    labels_train = labels_train_
    labels_test = labels_test_

    features_train_PCA = features_train
    features_test_PCA = features_test

    classifier = linear_model.LogisticRegression()

    # # extract features from data
    # pca = SPCA()
    # # PCA for training data
    # pca.fit(features_train)
    # print 'train features dimensions before PCA : ', features_train.shape
    # features_train_PCA = pca.transform(features_train)
    # print 'train features dimensions after PCA : ', features_train_PCA.shape
    #
    classifier.fit(features_train_PCA, labels_train)
    #
    # # PCA for test data features
    # print 'test features dimensions before PCA : ', features_test.shape
    # features_test_PCA = pca.transform(features_test)
    # print 'test features dimensions after PCA : ', features_test_PCA.shape

    # measure accuracy
    prediction = classifier.predict(features_test_PCA)
    accuracy = accuracy_score(prediction, labels_test)
    print 'accuracy given by logistic regression classifier: ', accuracy
    print '==============================================================================================='
    print 'features ', features_test.shape
    print 'labels ', len(labels_test)

    old_len = features_test.shape[1]
    h = w = 14
    new_features_test_array = features_test[:, :-(old_len-196)]
    print h, w, new_features_test_array.shape

    plot_gallery(new_features_test_array, labels_test, h, w)

    # plot the gallery of the most significative eigenfaces

    labels_PCA = [ i for i in labels_test]

    labels_PCA = np.asarray(labels_PCA)
    print 'labels_PCA ', labels_PCA.shape

    old_len = features_test_PCA.shape[1]
    eigenfaces = features_test_PCA[:, :-(old_len-196)]
    plot_gallery(eigenfaces, labels_PCA, h, w)

    plt.show()

    return accuracy
