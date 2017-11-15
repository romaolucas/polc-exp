# -*- coding: utf-8 -*-
'''
This module trains a given dataset (X, t) using logistic regression.
'''

import numpy as np

class LogRegClassifier:

    def __init__(self, X, t, thres=10**-4):
        self.X = X
        self.t = t
        self.thres = thres
        self.w = np.zeros(X.shape[1] + 1)

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
        err = err * - 1.0
        return err

    def err_grad(self):
        y = np.array([self.sigmoid(np.dot(x, self.w)) for x in self.X])
        grad = np.dot(self.X.T, (y - self.t))
        return grad 

    def calc_R_mat(self):
        Y = np.array([self.sigmoid(np.dot(x.T, self.w)) for x in self.X])
        R = np.diag(np.array([y*(1 - y) for y in Y]))
        return R

    '''
    Trains the model for logistic regression using
    the iterative reweighted least square methods (source: Bishop's book)
    '''
    def train(self):
        self.X = np.hstack((np.ones((self.X.shape[0], 1)), self.X))
        iterations = 0
        converged = False
        err = self.err_func()
        while not converged:
            R = self.calc_R_mat()
            R_inv = np.linalg.inv(R)
            Y = np.array([self.sigmoid(np.dot(x, self.w)) for x in self.X])
            z = np.dot(self.X, self.w) - np.dot(R_inv, (Y - self.t))
            # update w using the formula
            # w = (X'RX)^-1*X'Rz
            aux1 = np.matmul(self.X.T, R)
            aux2 = np.matmul(aux1, self.X)
            aux2 = np.linalg.inv(aux2)
            self.w = np.dot(aux2, np.dot(aux1, z))
            updated_err = self.err_func()
            print("err = {}, updated_err = {}, err - updated_err = {}".format(err, updated_err, err - updated_err))
            converged = (err - updated_err) < self.thres
            err = updated_err
            iterations += 1
        print("Convergiu apos {} iteracoes".format(iterations))

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        Y = np.array([self.sigmoid(np.dot(x, self.w)) for x in X])
        Y = np.array([y >= (1 - y) for y in Y]).astype(int)
        return Y

def calc_accuracy(Y, T):
    correctly_classified = 0
    for y, t in zip(Y, T):
        if y == t:
            correctly_classified += 1
    return correctly_classified / Y.shape[0]

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
classifier = LogRegClassifier(X_train, t_train)
classifier.train()
print("Rodando no conjunto de teste")
Y = classifier.predict(X_test)
print("Acuracia = ", calc_accuracy(Y, t_test))
