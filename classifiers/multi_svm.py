import numpy as np
import sklearn.datasets as datasets
import sklearn.model_selection as model_selection
from svm import SVMLearn
from kernelFactory import KernelFactory

class MultiSVMLearn(object):
    '''
    Implementation of SVM for multiple classes

    This implementation uses the OVA (one-versus-all) approach to solve
    the problem, i.e., the classification is made using K - 1 classifiers
    with K being the total number of classes.
    The class given to an input x is given by argmax_k y_k(x) with
    y_k(x) being computed by a SVMLearn classifier with y_k(x) > 0 if
    x is classified as k or y_k(x) if is classified as not k.

    Attributes:
    X: training set
    t: dictionary with K keys where t[k] is a list where the entry i is 1 if
    t_i = k, -1 otherwise
    kernel: kernel to be used in the classifiers. If none is given
    the base kernel is used, that is K(x, y) = x'y.
    classifiers: dictionary with K keys where classifiers[k] is the classifier
    for class k
    K: number of classes
    '''

    def __init__(self, classes, kernelType="", C=1, gamma=0.00001, degree=3, coef=0):
        '''
        Creates an instance of MultiSVMLearn
        '''
        self._classifiers = {}
        self._classes = classes
        for c in classes:
            self._classifiers[c] = SVMLearn(kernelType=kernelType, C=C, gamma=gamma, degree=degree, coef=coef)


    def fit(self, X, t):
        '''
        Trains the K classifiers for be used in train
        '''
        self._X = np.array(X)
        self._t = {}
        for c in self._classes:
            self._t[c] = np.array([1 if label == c else -1 for label in t])
        for k in self._classifiers:
            self._classifiers[k].fit(X, self._t[k])
    
    def train(self, X):
        y = []
        for x in X:
            y_x = {}
            for k in self._classifiers:
                y_x[k] = self._classifiers[k].compute_y(x)
            y.append(max(y_x, key=lambda k: y_x[k]))
        return y

    def compute_accuracy(self, X, t):
        y = self.train(X)
        correctly_classified = 0
        for y_i, t_i in zip(y, t):
            if y_i == t_i:
                correctly_classified += 1
        return correctly_classified / len(t)


def main():
    X, t = datasets.load_iris(return_X_y=True)
    X_train, X_test, t_train, t_test = model_selection.train_test_split(X, t)
    svm = MultiSVMLearn(set(t), kernelType="pol", C=1, degree=4)
    svm.fit(X_train, t_train)
    print("finished training")
    print("Accuracy: {}".format(svm.compute_accuracy(X_test, t_test)))


if __name__ == '__main__':
    main()
