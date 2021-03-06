__author__ = 'Oleksandra'

from sklearn.metrics import accuracy_score

from sklearn.utils import as_float_array

import numpy as np

from SPCA import SPCA
from SLR import SLR

from visualization import *

def classification_gradient_descent(features_train_, features_test_, labels_train_, labels_test_, labels_, dataset_):

    dataset = dataset_
    labels = labels_

    features_train = features_train_
    features_test = features_test_
    labels_train = labels_train_
    labels_test = labels_test_

    # extract features from data
    pca_2 = SPCA()
    # PCA for training data
    pca_2.fit(features_train)
    features_train_PCA_2 = pca_2.transform(features_train)
    features_test_PCA_2 = pca_2.transform(features_test)

    # thetas returned by PCA
    thetas = np.array(pca_2.S[:pca_2.k_components])
    # add dimension
    thetas = thetas[:, None]
    print 'theta dimensions : ', thetas.shape

    unique = np.unique(labels)
    n_classes = len(unique)
    print 'number of classes : ', n_classes

    ### training

    # create new training dataset with labels 1 or 0
    # 2d array where each column consists of labels for the corresponding class
    one_vs_all_labels_train = np.empty([len(features_train_PCA_2), n_classes], dtype=bool)
    print 'one_vs_all_labels dims : ', one_vs_all_labels_train.shape
    for current_class in range(n_classes):
        labels_class = np.copy(labels_test)
        print ' === ', labels_class
        for i, label in enumerate(labels_class):
            labels_class[i] = 1 if (label == current_class) else 0
        print '--- ', labels_class
        np.insert(one_vs_all_labels_train, current_class, labels_class)
    print 'c : '
    one_vs_all_labels = one_vs_all_labels_train.astype(int)
    print one_vs_all_labels

    # for every class create logistic regression classifier
    # train it via one-vs-all strategy
    log_reg_classifiers = np.empty([n_classes], dtype=type(SLR))
    for i, current_class in enumerate(log_reg_classifiers):
        print 'fitting thetas for classifier for class ', current_class
        classifier = SLR()
        current_labels = one_vs_all_labels_train[:, i]
        print 'current labels shape : ', current_labels.shape
        classifier.fit_thetas_grad_descent(thetas, features_train_PCA_2, current_labels)
        log_reg_classifiers[i] = classifier
        #np.insert(log_reg_classifiers, current_class, classifier)

    print 'all classifiers : ', len(log_reg_classifiers)

    ### testing

    # create new test dataset with labels 1 or 0
    # 2d array where each column consists of labels for the corresponding class
    one_vs_all_labels_test = np.empty([len(features_test_PCA_2), n_classes], dtype=bool)
    print 'one_vs_all_labels dims : ', one_vs_all_labels_test.shape

    # result matrix to store computed probabilities and select the highest one
    result_matrix = np.empty(one_vs_all_labels_test.shape, dtype=float)

    for current_class in range(n_classes):
        for i, clr in enumerate(log_reg_classifiers):
            pred = log_reg_classifiers[i].predict_thetas_prob(features_test_PCA_2, clr.thetas)
            np.insert(result_matrix, i, pred, axis=1)
            print 'i : ', i
            print pred
    """
    # cur_labels_to_predict = one_vs_all_labels_test[:, current_class]
    sum_accuracy = 0

    for class_ in range(n_classes):
        current_class_labels = one_vs_all_labels[:, class_]
        print '================================================================'
        print current_class_labels
        accuracy = accuracy_score(pred, current_class_labels)
        sum_accuracy += accuracy
        print 'for class ', class_, ' accuracy : ', accuracy

    print 'average accuracy : ', sum_accuracy/n_classes
    # measure accuracy
    # prediction = classifier.predict(features_test_PCA_2)
    # accuracy = accuracy_score(prediction, labels_test)
    # print 'accuracy given by logistic regression classifier: ', accuracy

    faces_images = dataset.images
    print_faces(faces_images, labels, 9)

    return sum_accuracy/n_classes"""
    return 0
