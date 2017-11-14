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
    def create_kernel(kernel_name, *args, **kwargs):
        if kernel_name == 'pol':
            return PolynomialKernel(*args, **kwargs)
        if kernel_name == 'rbf':
            return RBFKernel(*args, **kwargs)
        if kernel_name == 'laplace':
            return LaplacianKernel(*args, **kwargs)
        if kernel_name == 'tanh':
            return HyperbolicTangentKernel(*args, **kwargs)
        return BaseKernel(*args, **kwargs)
