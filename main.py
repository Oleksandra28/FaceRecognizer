__author__ = 'Oleksandra'


from sklearn.datasets import fetch_olivetti_faces
from sklearn.cross_validation import  train_test_split

from classification_SVD_values import classification_SVD_values
from classification_log_reg import classification_log_reg
from classification_gradient_descent import classification_gradient_descent

from sklearn.metrics import accuracy_score

from sklearn.utils import as_float_array
from sklearn import linear_model

import matplotlib.pyplot as plt

import numpy as np

from SPCA import SPCA
from SLR import SLR


#########################################################################################################
if __name__ == "__main__":

    dataset = fetch_olivetti_faces()
    data = dataset.data
    labels = dataset.target

    print 'dataset data dimensions : ', data.shape
    print 'dataset labels dimensions : ', labels.shape

    # TODO print eigenfaces normally!!!!
    #n_samples, h, w = dataset.images.shape
    #faces_images = dataset.images
    #print_faces(faces_images, labels, 20)

    # split dataset for training and evaluation
    test_percent = 0.3
    features_train, features_test, labels_train, labels_test = train_test_split(data, labels, test_size = test_percent)

    #########################################################################################################

    #accuracy_SVD = classification_SVD_values(features_train, features_test, labels_train, labels_test, labels, dataset)
    #accuracy_log_reg = classification_log_reg(features_train, features_test, labels_train, labels_test, labels, dataset)
    accuracy_grad_descent = classification_gradient_descent(features_train, features_test, labels_train, labels_test, labels, dataset)

    print '##################################################################################'
    #print 'accuracy using SVD weights : ', accuracy_SVD
    #print 'accuracy using log_reg     : ', accuracy_log_reg
    print 'accuracy using grad descent: ', accuracy_grad_descent






