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
        z = round(z, 9)
        exponent = Decimal(np.exp(z))
        denominator = (1 + exponent)
        result = Decimal(1)/Decimal(denominator)

        # convert result to prevert overflow and underflow
        min = Decimal(0.000000001)
        max = Decimal(0.999999999)
        if result.compare(min) == Decimal('-1'):
            result = min
        elif result.compare(max) == Decimal('1'):
            result = max

        #result = np.nan_to_num(result)

        result = round(result, 9)
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
    def fit_thetas_grad_descent(self, initial_thetas, features_train, labels_train, learning_rate = 0.001, n_iters = 10):
        thetas = initial_thetas
        m = features_train.shape[0]

        current_thetas = np.around(thetas, decimals=2)
        for i in range(n_iters):
            print 'iteration ', i
            # compute cost of a particular choice of theta

            predictions = self.predict_thetas_prob(features_train, current_thetas)
            current_cost = self.compute_cost_vectorized(predictions, labels_train)

            # compute gradient
            grad = np.multiply(np.dot(np.transpose(features_train), (predictions-labels_train)), 1./m)
            grad = np.multiply(grad, learning_rate)

            lambda_ = 1.
            regularization_term = sum(np.multiply(pow(current_thetas, 2.), lambda_/m))
            current_thetas = np.add(np.subtract(current_thetas, grad), regularization_term)
        #end for loop

        #print 'thetas after grad descent : ', thetas
        self.set_thetas(thetas)

#------------------------------------------------------------------------------------------------------------------
    def compute_cost_vectorized(self, predicted_labels, real_labels):

        predicted_labels = np.around(predicted_labels, decimals=9)

        m = predicted_labels.shape[0]

        positive_term = np.multiply(real_labels, np.log(predicted_labels))

        min = Decimal(0.000000001)
        max = Decimal(0.999999999)

        for i, p in enumerate(positive_term):
            temp  = Decimal(str(p[0]))
            if temp.compare(min) == Decimal('-1'):
                positive_term[i] = min
            elif temp.compare(max) == Decimal('1'):
                positive_term[i] = max
            positive_term[i] = round(positive_term[i], 9)

        positive_term = np.around(positive_term, decimals=9)
        #positive_term = np.nan_to_num(positive_term)
        negative_term = np.multiply((1-real_labels), np.log(1- predicted_labels))

        for i, p in enumerate(negative_term):
            temp  = Decimal(str(p[0]))
            if temp.compare(min) == Decimal('-1'):
                negative_term[i] = min
            elif temp.compare(max) == Decimal('1'):
                negative_term[i] = max
            negative_term[i] = round(negative_term[i], 9)

        negative_term = np.around(negative_term, decimals=9)

        #negative_term = np.nan_to_num(negative_term)
        total_error_sum = np.add(positive_term, negative_term)
        print 'total_error_sum = ', sum(total_error_sum)

        j = np.around(np.multiply(total_error_sum, (-(1./m))),decimals=9)

        #cost = np.sum(j)
        #print 'cost ', cost
        cost = j
        return cost

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
        #prediction = np.around(prediction, decimals=9)
        #print 'prediction calculation checked. parent commit 5a236802b653ea288abcc90d844567a2244683f3'
        return prediction

#------------------------------------------------------------------------------------------------------------------
