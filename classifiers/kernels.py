import numpy as np
import math

class BaseKernel():
    '''
    The BaseKernel implementation will be the identify kernel, that is,
    K(x, y) = x'y
    '''
    
    def __init__(self, *args, **kwargs):
        pass

    def apply(self, x, y):
        return np.dot(x, y)

class PolynomialKernel(BaseKernel):
    '''
    The PolynomialKernel will take a parameter d and an optional parameter c, if none is given
    c is 0
    K(x, y) = (x'y + c)^d
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d = args[0]
        if len(args) > 1:
            self.c = args[1]
        else:
            self.c = 0

    def apply(self, x, y):
        return math.pow((np.dot(x, y) + self.c), self.d)

class RBFKernel(BaseKernel):
    '''
    The RBFKernel takes a single parameter gamma. 
    K(x, y) = exp(-gamma*d(x, y)^2)
    with d being the euclidean distance from x to y
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = args[0]

    def apply(self, x, y):
        distance = np.linalg.norm(x - y)
        return np.exp(-1 * self.gamma*math.pow(distance, 2))

class LaplacianKernel(BaseKernel):
    '''
    The LaplacianKernel takes alpha as a parameter.
    K(x, y) = exp(-alpha*d(x, y))
    with d being the euclidean distance from x to y
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = args[0]

    def apply(self, x, y):
        distance = np.linalg.norm(x - y)
        return np.exp(-1 * self.alpha*distance)

class HyperbolicTangentKernel(BaseKernel):
    '''
    The HyperbolicTangent Kernel takes two parameters:
    alpha and c. If no value for c is given, c equals 0.
    K(x, y) = tanh(alpha*x'y + c)
    '''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = args[0]
        if len(args) > 1:
            self.c = args[1]
        else:
            self.c = 0

    def apply(self, x, y):
        return np.tanh(self.alpha*np.dot(x, y) + self.c)
