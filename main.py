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

def print_faces(images, target, top_n):
    print 'faces : '
    # set up the figure size in inches
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(top_n):
        # plot the images in a matrix of 20x20
        p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)

        # label the image with the target value
        p.text(0, 14, str(target[i]))
        p.text(0, 60, str(i))
#---------------------------------------------------------------------------------------------------------

def plot_gallery(images_, titles_, h, w, n_row=3, n_col=3):
    images = np.array(images_)
    titles = np.array(titles_)

    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        image = images[i]
        image = image.squeeze()
        image = image.reshape((h, w))
        plt.imshow(image, cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

#########################################################################################################
if __name__ == "__main__":

    dataset = fetch_olivetti_faces()
    classifier = linear_model.LogisticRegression()

    test_percent = 0.3
    data = dataset.data
    labels = dataset.target
    print 'dataset data dimensions : ', data.shape
    print 'dataset labels dimensions : ', labels.shape

    n_samples, h, w = dataset.images.shape
    faces_images = dataset.images
    #print_faces(faces_images, labels, 20)

    # split dataset for training and evaluation
    features_train, features_test, labels_train, labels_test = train_test_split(data, labels, test_size = test_percent)
    """
    # extract features from data
    pca = SPCA()
    # PCA for training data
    pca.fit(features_train)
    print 'train features dimensions before PCA : ', features_train.shape
    features_train_PCA = pca.transform(features_train)
    print 'train features dimensions after PCA : ', features_train_PCA.shape

    classifier.fit(features_train_PCA, labels_train)

    # PCA for test data features
    print 'test features dimensions before PCA : ', features_test.shape
    features_test_PCA = pca.transform(features_test)
    print 'test features dimensions after PCA : ', features_test_PCA.shape

    # measure accuracy
    prediction = classifier.predict(features_test_PCA)
    accuracy = accuracy_score(prediction, labels_test)
    print 'accuracy given by logistic regression classifier: ', accuracy
    print '==============================================================================================='

    print 'features_test ', features_test.shape, h, w
    print 'labels_test ', labels_test.shape

    plot_gallery(features_test, labels_test, h, w)

    # plot the gallery of the most significative eigenfaces

    labels_PCA = ["%d" % i for i in range(features_test_PCA.shape[0])]
    print 'features_test_PCA ', features_test_PCA.shape
    print 'labels_PCA ', len(labels_PCA)
    #eigenfaces = np.array(features_test_PCA)
    #eigenfaces = eigenfaces.reshape((pca.k_components, h/8, w/8))
    #plot_gallery(eigenfaces, labels_PCA, h/8, w/8)
    plt.show()
    """
#########################################################################################################

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

    # for every class create logistic regression classifier
    # train it via one-vs-all strategy
    log_reg_classifiers = np.empty([n_classes])

    for current_class in range(n_classes):
        print 'fitting thetas for classifier for class ', current_class
        classifier = SLR()
        classifier.fit_thetas(thetas)
    print 'all classifiers : ', len(log_reg_classifiers)

    ### testing

    # create new test dataset with labels 1 or 0
    # 2d array where each column consists of labels for the corresponding class
    one_vs_all_labels = np.empty([len(features_test_PCA_2), n_classes], dtype=int)
    print 'one_vs_all_labels dims : ', one_vs_all_labels.shape
    for current_class in range(n_classes):
        labels_class = np.copy(labels_test)
        for i, label in enumerate(labels_class):
            labels_class[i] = 1 if (label == current_class) else 0
        np.insert(one_vs_all_labels, current_class, labels_class)
    print 'one_vs_all_labels : '
    print one_vs_all_labels
    max_pred = -1
    best_estimator  = -1
    result_prediction = np.empty([len(labels_test)])
    # for each classifier, find the most probable class
    # for i, classifier in enumerate(log_reg_classifiers):
    #     pred = classifier.predict(features_test_PCA_2)
    #     print 'pred : ', pred
    #     if pred > max_pred:
    #         best_estimator = i

    cls = log_reg_classifiers[0]
    pred = classifier.predict(features_test_PCA_2)

    for class_ in range(n_classes):
        current_class_labels = one_vs_all_labels[:, class_]
        print '================================================================'
        print current_class_labels
        accuracy = accuracy_score(pred, current_class_labels)
        print 'for class ', class_, ' accuracy : ', accuracy
    # measure accuracy
    # prediction = classifier.predict(features_test_PCA_2)
    # accuracy = accuracy_score(prediction, labels_test)
    # print 'accuracy given by logistic regression classifier: ', accuracy
    print '==============================================================================================='













