from kernels import *

class KernelFactory():
    '''
    Basic implementation of the factory pattern
    given a string value it will create the 
    corresponding kernel. If the required kernel
    takes any argument, it's assumed to be passed
    to the method create_kernel, otherwise it will
    return an exception
    '''

    @staticmethod
    def create_kernel(kernel_name, gamma, degree, coef):
        if kernel_name == 'pol':
            return PolynomialKernel(gamma, degree, coef)
        if kernel_name == 'rbf':
            return RBFKernel(gamma, degree, coef)
        if kernel_name == 'laplace':
            return LaplacianKernel(gamma, degree, coef)
        if kernel_name == 'tanh':
            return HyperbolicTangentKernel(gamma, degree, coef)
        return BaseKernel(gamma, degree, coef)
