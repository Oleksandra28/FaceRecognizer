__author__ = 'Oleksandra'

import numpy as np

class SLR():

    def __init__(self):
        self.sigmoid = np.vectorize(self.sigmoid)
        #self._convert_prediction = np.vectorize(self._convert_prediction)
#------------------------------------------------------------------------------------------------------------------
    def sigmoid(self, z):
        result = float(1./float((1 + np.exp(-z))))
        print 'sigmoid : ', result
        return result
#------------------------------------------------------------------------------------------------------------------
    def fit_thetas(self, thetas):
        print 'given thetas dimensions : ', thetas.shape
        self.thetas = thetas
        #print 'thetas : ', thetas
        #self.thetas = np.transpose(thetas)
        #print 'thetas transposed dimensions : ', self.thetas.shape
#------------------------------------------------------------------------------------------------------------------
    def predict(self, features):
        z = np.dot(features, self.thetas)
        print 'z dimensions : ', z.shape
        prediction = self.sigmoid(z)
        print 'prediction : ', prediction.shape

        prediction = prediction.astype(int)
        print prediction
        return prediction
        #return self._convert_prediction(prediction)
#------------------------------------------------------------------------------------------------------------------
    def _convert_prediction(self, prediction, threshold = 0.5):
        result = np.empty(len(prediction), dtype=bool)
        for i, p in enumerate(prediction):
            np.insert(result, i, bool(p >= threshold))
        return result



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
