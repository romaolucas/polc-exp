# -*- coding: utf-8 -*-
'''
This module trains a given dataset (X, t) using logistic regression.
'''

import numpy as np
from sklearn.base import BaseEstimator

class LogRegClassifier(BaseEstimator):

    def __init__(self, lamb=1e-4, thres=10**-4):
        self.lamb = lamb
        self.thres = thres

    '''
    calculates sigmoid for given value in R
    '''
    def sigmoid(self, a):
        return 1./( 1 + np.exp(-a))


    '''
    calculates error function given by the formula:
        - sum_{i = 1}^{N} [t_n ln(y_n) + (1 - t_n)ln(1 - y_n)]
    '''
    def err_func(self):
        N = self.X.shape[0]
        err = 0
        for i in range(0, N):
            y = self.sigmoid(np.dot(self.w, self.X[i]))
            err = err + (self.t[i]*np.log(y) + (1 - self.t[i])*np.log(1 - y))
        err -= 0.5*self.lamb*np.dot(self.w, self.w)
        err = err * - 1.0
        return err

    def err_grad(self):
        y = np.array([self.sigmoid(np.dot(x, self.w)) for x in self.X])
        grad = np.dot(self.X.T, (y - self.t)) + self.lamb*self.w
        return grad 

    def calc_R_mat(self):
        Y = np.array([self.sigmoid(np.dot(x.T, self.w)) for x in self.X])
        R = np.diag(np.array([y*(1 - y) for y in Y]))
        return R

    '''
    Trains the model for logistic regression using
    the iterative reweighted least square methods (source: Bishop's book)
    '''
    def fit(self, X, t):
        self.X = X
        self.t = t
        self.X = np.hstack((np.ones((self.X.shape[0], 1)), self.X))
        self.w = np.zeros(X.shape[1] + 1)
        n, m = self.X.shape
        iterations = 0
        converged = False
        err = self.err_func()
        while not converged:
            R = self.calc_R_mat()
            grad = self.err_grad()
            XR = np.dot(self.X.T, R)
            Hessian = np.dot(XR, self.X) + self.lamb*np.identity(m)
            self.w = self.w - np.linalg.solve(Hessian, grad)
            updated_err = self.err_func()
            converged = (err - updated_err) < self.thres
            err = updated_err
            iterations += 1
        print("Convergiu apos {} iteracoes".format(iterations))

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        Y = np.array([self.sigmoid(np.dot(x, self.w)) for x in X])
        Y = np.array([y >= (1 - y) for y in Y]).astype(int)
        return Y

    def score(self, X, t):
        y = self.predict(X)
        correctly_classified = 0
        for y_i, t_i in zip(y, t):
            if y_i == t_i:
                correctly_classified += 1
        return correctly_classified / y.shape[0]

def main():
    X = []
    t = []
    import csv
    with open('clean2.data', mode='r') as csvfile:
        data_reader = csv.reader(csvfile, delimiter=',')
        for row in data_reader:
            X.append(row[0:166])
            t.append(row[-1])

    X = np.array(X, dtype=float)
    t = np.array(t, dtype=float)

    from sklearn.model_selection import train_test_split
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.33)

    print("Treinando modelo")
    classifier = LogRegClassifier()
    classifier.fit(X_train, t_train)
    print("Rodando no conjunto de teste")
    print("Acuracia = ", classifier.score(X_test, t_test))

if __name__ == "__main__":
    main()
