import numpy as np
from Activation import Activation

class HiddenLayer(object):    
    def __init__(self,n_in, n_out,
                 activation_last_layer='relu',activation='relu', W=None, b=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        
        bias vector b -> 

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: string
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input=None
        self.activation=Activation(activation).f
        
        # activation deriv of last layer
        self.activation_deriv=None
        if activation_last_layer:
            self.activation_deriv=Activation(activation_last_layer).f_deriv

        # we randomly assign small values for the weights as the initiallization
        # lucas comment - weight init
        

        
        # print(n_in) equals to 128
        # print(n_out) equals to 3
        # exit()
        self.W = np.random.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
        )

     

        if activation == 'relu':
            self.W *= 4

        # we set the size of bias as the size of output dimension
        self.b = np.zeros(n_out,)

        self.weight_V_t = np.zeros(self.W.shape)
        self.bias_V_t = np.zeros(self.b.shape)
        
        # we set he size of weight gradation as the size of weight
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)
        
    
    # the forward and backward progress for each training epoch
    # please learn the week2 lec contents carefully to understand these codes. 
    def forward(self, input):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)a
        '''
        #np.dot - product of two array
        print('----------')
        # print(input.shape)
        # print(self.W.shape)
        # print(self.b.shape)
        print('----------')
        lin_output = np.dot(input, self.W) + self.b
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )


        # do one hot encoding
        # if self.activation.__name__ == '__softmax':
        #     one_hot_encode = np.zeros(len(self.output))

        #     largest_prob = np.amax(self.output)

        #     largest_prob_index = np.where(self.output == largest_prob)

        #     one_hot_encode[largest_prob_index] = 1

        #     print('ya')

        # else:
        #     one_hot_encode = None    

        self.input=input

        print(self.activation.__name__)
            
        return self.output
    
    def backward(self, delta, output_layer=False):
       
        #.T means in transpose it
        # delta

        # grad_W is J(theta) - which is J(W) - gradient of weight
        # grad_b is J(theta) - which is J(b) - gradient of bias

        # delta term is for layer 2,3,4 - there is no delta term for layer 1

        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = delta
        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
        return delta