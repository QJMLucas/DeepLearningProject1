import numpy as np

class Activation(object):
    
    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        # a = np.tanh(x)   
        return 1.0 - a**2
    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_deriv(self, a):
        # a = logistic(x) 
        return  a * (1 - a )
    
    def __softmax(self, x):
        # return np.exp(x) / np.sum(np.exp(x), axis =  0)
        print(x)
        return np.exp(x)/np.sum(np.exp(x),axis=-1)

    def __softmax_deriv(self, x):
        return x * (1 - x)
    
    def __relu(self, x):
        
        return np.maximum(0,x)
    
    def __relu_deriv(self, x):

        x[x<=0] = 0
        x[x>0] = 0
    
        return x
    
    def __init__(self, activation='tanh'):
        
        if activation == 'logistic':
            self.f = self.__logistic
            self.f_deriv = self.__logistic_deriv
        elif activation == 'tanh':
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv
        elif activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_deriv
        elif activation == 'softmax':
            self.f = self.__softmax
            self.f_deriv = self.__softmax_deriv