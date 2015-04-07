__author__ = 'Oleksandra'

import numpy as np

class SLR():

    def __init__(self):
        self.sigmoid = np.vectorize(self.sigmoid)
        #self._convert_prediction = np.vectorize(self._convert_prediction)
#------------------------------------------------------------------------------------------------------------------
    def sigmoid(self, z):
        result = float(1./float((1 + np.exp(-z))))
        #print 'sigmoid : ', result
        return result
#------------------------------------------------------------------------------------------------------------------
    def fit_thetas(self, thetas):
        #print 'given thetas dimensions : ', thetas.shape
        self.thetas = thetas
        #print 'thetas : ', thetas
        #self.thetas = np.transpose(thetas)
        #print 'thetas transposed dimensions : ', self.thetas.shape
#------------------------------------------------------------------------------------------------------------------
    def predict(self, features):

        z = np.dot(features, self.thetas)
        prediction = self.sigmoid(z)

        return self._convert_prediction(prediction)
#------------------------------------------------------------------------------------------------------------------
    def _convert_prediction(self, prediction, threshold = 0.5):
        result = np.empty(len(prediction), dtype=bool)
        for i, p in enumerate(prediction):
            np.insert(result, i, bool(p >= threshold))
        print 'result :', result.shape
        result = result.astype(int)
        print result
        return result
#------------------------------------------------------------------------------------------------------------------
    # calculate parameters theta using gradient descent
    def fit_thetas_grad_descent(self, initial_thetas, features_train, labels_train, learning_rate = 0.01, n_iters = 100):
        thetas = initial_thetas
        m = features_train.shape[0]
        print 'm = ', m
        j_history = np.empty([m, n_iters])

        for i in range(n_iters):

            # compute cost of a particular choice of theta
            predictions = self.predict_thetas(features_train, thetas)

            current_cost = self.compute_cost_vectorized(predictions, labels_train)
            print 'current_cost : ', current_cost.shape
            print 'j_his', j_history.shape
            # save the cost J in every iteration
            #np.insert(j_history, i, current_cost)#, axis=1)
            j_history[:, i] = current_cost

            # compute gradient
            grad = np.multiply(np.dot(np.transpose(features_train), (predictions-labels_train)), 1/m)
            print 'grad: ', grad.shape
            grad = grad[:, None]
            thetas -= grad*learning_rate

        print 'thetas after grad descent : ', thetas
        self.fit_thetas(thetas)

#------------------------------------------------------------------------------------------------------------------
    def compute_cost_vectorized(self, predicted_labels, real_labels):

        #predicted_labels = predicted_labels[:, None]
        print 'predicted_labels : ', predicted_labels.shape
        print predicted_labels

        #real_labels = real_labels[:, None]
        print 'real_labels : ', real_labels.shape
        real_labels = np.asarray(real_labels, dtype=int)
        print real_labels

        m = predicted_labels.shape[0]
        #total_error_sum = (np.multiply(real_labels, np.log(predicted_labels)) + np.multiply((1-real_labels), np.log(1- predicted_labels)))
        total_error_sum = np.add(np.multiply(real_labels, np.log(predicted_labels)), np.multiply((1-real_labels), np.log(1- predicted_labels)))
        print 'total_error_sum ', total_error_sum.shape
        j = np.around(np.multiply(total_error_sum, (-1/m)),decimals=3)

        return j

#------------------------------------------------------------------------------------------------------------------
    def predict_thetas(self, features, thetas):
        z = np.dot(features, thetas)
        prediction = self.sigmoid(z)
        return self._convert_prediction(prediction)

#------------------------------------------------------------------------------------------------------------------
    def predict_thetas_prob(self, features, thetas):
        z = np.dot(features, thetas)
        prediction = self.sigmoid(z)
        prediction = np.around(prediction, decimals=3)
        print 'prediction : ', prediction
        return prediction

#------------------------------------------------------------------------------------------------------------------
    # # softmax function for multi class logistic regression
    # def softmax(self, W,b,x):
    #    vec=np.dot(x,W.T)
    #    vec=np.add(vec,b)
    #    vec1=np.exp(vec)
    #    res=vec1.T/np.sum(vec1,axis=1)
    #    return res.T
    #
    # """ function predicts the probability of input vector x"""
    # """ the output y is MX1 vector (M is no of classes) """
    # def predict_multiclass(self,x):
    #     y = self.softmax(self.W,self.b,x)
    #     return y
    #
    # ''' function returns the lables corresponding to the input y '''
    # def label(self,y):
    #     return self.labels[y];
    #
    # ''' function classifies the input vector x into one of output lables '''
    # ''' input is NXD vector then output is NX1 vector '''
    #
    # def classify_multiclass(self,x):
    #     result=self.predict(x);
    #     indices=result.argmax(axis=1);
    #     #converting indices to labels
    #     labels=map(self.label, indices);
    #     return labels
    #
