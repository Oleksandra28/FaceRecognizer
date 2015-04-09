__author__ = 'Oleksandra'

import numpy as np

from decimal import Decimal

class SLR():

    def __init__(self):
        self.sigmoid = np.vectorize(self.sigmoid)
        #self._convert_prediction = np.vectorize(self._convert_prediction)
#------------------------------------------------------------------------------------------------------------------
    def sigmoid(self, z):
        z = Decimal(str(-z))
        exponent = Decimal(np.exp(z))
        denominator = (1 + exponent)
        result = Decimal(1)/Decimal(denominator)
        print 'sigmoid : ', result
        # convert result to prevert overflow and underflow
        min = Decimal(0.000000001)
        max = Decimal(0.999999999)
        if result.compare(min) == Decimal('-1'):
            result = max
        elif result.compare(max) == Decimal('1'):
            result = min
        result = round(result, 9)
        #result = float(result.to_eng_string)
        print 'result converted : ', result
        return result
#------------------------------------------------------------------------------------------------------------------

    def set_thetas(self, thetas):
        self.thetas = thetas

#------------------------------------------------------------------------------------------------------------------
    def predict(self, features):

        z = np.dot(features, self.thetas)
        prediction = self.sigmoid(z)

        return self._convert_prediction(prediction)
#------------------------------------------------------------------------------------------------------------------
    def _convert_prediction(self, prediction, threshold = 0.5):
        #result = np.empty(prediction.shape, dtype=int)
        result = None
        #TODO vectorize
        for i, p in enumerate(prediction):
            val = p+threshold
            print 'val before ', val
            val = np.floor(val)
            print 'val after ', val
            result = np.insert(result, i, val)#np.isclose(p, threshold))#bool(p >= threshold))
        print 'result before convertion:', result.shape
        print result
        print 'result vector converted ',result.shape
        result.squeeze()
        print result
        return result
#------------------------------------------------------------------------------------------------------------------
    # calculate parameters theta using gradient descent
    def fit_thetas_grad_descent(self, initial_thetas, features_train, labels_train, learning_rate = 0.01, n_iters = 100):
        thetas = initial_thetas
        m = features_train.shape[0]
        j_history = np.empty([m, n_iters])

        for i in range(n_iters):

            # compute cost of a particular choice of theta
            #predictions = self.predict_thetas(features_train, thetas)

            thetas = np.around(thetas, decimals=2)
            predictions = self.predict_thetas_prob(features_train, thetas)

            current_cost = self.compute_cost_vectorized(predictions, labels_train)

            # save the cost J in every iteration
            current_cost = current_cost.squeeze()
            j_history[:, i] = current_cost

            # compute gradient
            grad = np.multiply(np.dot(np.transpose(features_train), (predictions-labels_train)), 1/m)
            thetas = thetas - np.multiply(grad, learning_rate)
        #end for loop

        #print 'thetas after grad descent : ', thetas
        self.set_thetas(thetas)

#------------------------------------------------------------------------------------------------------------------
    def compute_cost_vectorized(self, predicted_labels, real_labels):

        predicted_labels = np.around(predicted_labels, decimals=3)

        m = predicted_labels.shape[0]

        positive_term = np.multiply(real_labels, np.log(predicted_labels))
        print 'pos term', positive_term
        positive_term = np.around(positive_term, decimals=9)

        negative_term = np.multiply((1-real_labels), np.log(1- predicted_labels))
        print 'neg term', negative_term
        negative_term = np.around(negative_term, decimals=9)

        total_error_sum = np.add(positive_term, negative_term)

        j = np.around(np.multiply(total_error_sum, (-(1./m))),decimals=3)

        #print 'cost after ', j
        return j

#------------------------------------------------------------------------------------------------------------------
    def predict_thetas(self, features, thetas):
        z = np.dot(features, thetas)
        prediction = self.sigmoid(z)
        prediction = np.around(prediction, decimals=0)
        return prediction
        # weird conversion goes on here
        #return self._convert_prediction(prediction)

#------------------------------------------------------------------------------------------------------------------
    def predict_thetas_prob(self, features, thetas):
        thetas = np.around(thetas, decimals=4)
        z = np.dot(features, thetas)
        prediction = self.sigmoid(z)
        prediction = np.around(prediction, decimals=9)
        print 'prediction calculation checked. parent commit 5a236802b653ea288abcc90d844567a2244683f3'
        return prediction

#------------------------------------------------------------------------------------------------------------------
