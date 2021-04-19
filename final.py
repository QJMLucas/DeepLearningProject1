import numpy as np
from numpy.core import defchararray
# import matplotlib.pyplot as plt

EPOCH = 500
n_batchsize = 5
dropout_ratio = 0.2
learning_rate = 0.001
momentum = 0.8
decay = 0.01
eps = 1e-5
W_2 = 0

# partial derivative is gradient
class Layer(object):

    def __init__(self, n_in, n_out, W = None, b = None, learning_rate=learning_rate, momentum=momentum, decay=decay):

        if W:
            self.W = W
        else:
            # randomly assign small values for the weights as the initiallization
            self.W = np.random.uniform(
                low=-np.sqrt(6.0 / (n_in + n_out)),
                high=np.sqrt(6.0 / (n_in + n_out)),
                size=(n_in, n_out)
            )
            
        if b:
            self.b = b
        else:
            self.b = np.zeros([1, n_out])

        # print(self.b.shape) # shape (1,256)
    
        self.n_in = n_in
        self.n_out = n_out

        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

        self.learning_rate = learning_rate

        # momentum
        self.momentum = momentum

        # v in Momentum
        self.v_W = np.zeros(self.W.shape)
        self.v_b = np.zeros(self.b.shape)

        # weight decay
        self.decay = decay

    def forward(self, input): # numpy.array

        # input size is [batch_size, n_in]
        # shape input: [batchsize, n_in]
        # shape return: [batchsize, n_out]

        self.input = input

        output = (
            
            # this part can be adjusted to use original code
            # np.dot(input, self.W) + self.b
            input.dot(self.W) + self.b
        )#[batch_size, n_in] times [x_in, n_out] + [1,n_out] = [bach_size, n_out]

        global W_2
 
        W_2 += (self.W ** 2).sum()

        return output

    def backward(self, delta):
        
        delta = delta.dot(

            self.W.T
        ) # [batch_size, n_out] times [n_in, n_out].T = [batchsize, n_in]

        return delta
    
    def update(self, delta):
        self.grad_W = (
            np.atleast_2d(self.input).T.dot(np.atleast_2d(delta)) / n_batchsize
        ) # [batchsize, n_in]. T times [batchsize, n_out] = [n_in, n_out]

        # print(self.input.T.shape)
        # print(delta.shape)
        # print(self.grad_W.shape)

        self.grad_b = delta.mean(axis = 0)

        # print(self.v_W.shape)
        # print(self.grad_W.shape)
        
        # debug - errors here - self.v_W.shape: shape(128,256)
        # 
        # self.grad_W.shape - shape(128,128)


        self.v_W = (
            self.momentum * self.v_W + self.learning_rate * self.grad_W
        ) # [n_in, n_out] + [n_in, n_out] -> [n_in, n_out]

        self.W = (1- self.learning_rate * self.decay) * self.W - self.v_W

        self.v_b = (
            self.momentum * self.v_b + self.learning_rate * self.grad_b
        )

        self.b -= self.v_b


# relu

class ReLU(object):
    def __init__(self):
        self.input = None

    def forward(self, input): 
        self.input = input    #[batchsize, input]
        # print(input.shape)   #[5,256] - seems not correct because it should be batchsize, input
                             # batch size is correct 5, input is 256?
        
        return np.maximum(0, input)

    def backward(self, delta):
        # print('testing')
        # print(delta.shape) # (5,256) - correct

        # result is zero when value is less than 1
        delta[self.input<=0] = 0 #delta shape [batchsize, n_out]

        return delta


class Dropout(object):
    def __init__(self, ratio = dropout_ratio):
        self.ratio = ratio

    def forward(self, input, is_testing=False):
        if is_testing:
            return input

        else:
            
            self.r = input * (np.random.rand(input.shape[0], input.shape[1]) > self.ratio)
            dropout_result = input * (self.r > self.ratio)
            return dropout_result


    def backward(self, delta):
        # shape delta [n_batchsize, n_out]

        delta[self.r < self.ratio] = 0

        return delta


# batch normalisation does not change the shape of the input

# epsilon is added to prevent 0 occuring
class BatchNormalisation(object):
    def __init__(
        self,
        n_in, #int
        momentum,
        learning_rate = learning_rate,
        eps = eps
    ):

    # gets the mean and variance for the batch

        self.mean = 0.0
        self.var = 10.0
        self.eps = eps

        # beta for shift
        self.beta = np.zeros([1, n_in])

        # gamma for scale
        self.gamma = np.ones([1, n_in])
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v_gamma = np.zeros([1, n_in])
        self.v_beta = np.zeros([1, n_in])

    def forward(self, input, is_testing=False): # input numpy array -> "numpy.array"
        
        self.input = input
        # if it's testing 
        if is_testing:
            scale = self.gamma/ np.sqrt(self.var * self.eps)
            out = input * scale + (self.beta - self.mean * scale)
            return out
        
        # axis=0 -> to get the mean and variance verically
        # mean and variance for per batch per feature
        self.input_mean = input.mean(axis=0)
        self.input_var = input.var(axis=0)

        # keep the overall mean and variance
        # gamma(momentum) + (1-gamma) * input_mean
        self.mean = self.momentum * self.mean + (1 - self.momentum) * self.input_mean
        self.var = self.momentum * self.var + (1 - self.momentum) * self.input_var


        # normalise step 
        self.x_hat = (input - self.input_mean) / np.sqrt(self.input_var + self.eps)
        
        # scale and shift
        # gamma times x_hat + beta
        self.y=self.gamma * self.x_hat + self.beta
        

        return self.y

    # delta - derivative 
    # previous derivative times derivative of the current layer and send backward
    def backward(self, delta): # delta - numpy array -> outputs numpy array
 
    # In batch normalisation, n_in = n_out)
    # gamma is shape [1, n_out]
    # delta is shape [n_batchsize, n_out]
    # so the shape of grad_x_hat is still [n_batchsize, n_out]
                
        grad_x_hat = (self.gamma * delta)


    # [n_batchsize, n_out] * [n_batchsize, n_in]
    # the sum function results in [1, n_out]
    # n_out and n_in are the same
        grad_sigma_2 = (
            -0.5 
            * (grad_x_hat * (self.input - self.input_mean)).sum(
                axis=0) * np.power(self.input_var + eps, -1.5))

        grad_mu = (
            (grad_x_hat / np.sqrt(self.input_var + eps)).sum(
                axis=0
            )
            -2
            * grad_sigma_2
            * (self.input- self.input_mean).sum(
                axis = 0
            )
            /n_batchsize
        )

        grad_x = (
            grad_x_hat / np.sqrt(self.input_var + eps)
            +2.0
            * grad_sigma_2
            * (self.input - self.input_mean)
            / n_batchsize
            + grad_mu / n_batchsize
        )

        return grad_x


# output layer
# input of softmax is x = [x0, x1, ... xn]
# shape for the input is the same as the shape of the output
class Softmax(object):
    def __init__(self):
        pass

    def forward(self, input): #input - numpy array -> outputs numpy.array
        self.output = np.apply_along_axis(
            lambda x: np.exp(x) / np.exp(x).sum(), axis = 1, arr = np.atleast_2d(input)
        )
    
        return self.output

    def backward(self, delta):
        grad_x = []

        for i in range(n_batchsize):
            grad_x_ = np.diag(self.output[i]) - np.outer(
                self.output[i], self.output[i]
            )

            grad_x.append(
                np.atleast_2d(delta[i]).dot(grad_x_)
            )
        
        grad_x = np.stack(
            grad_x, axis = 0
        ).squeeze()

        return grad_x

class CrossEntropyLoss(object):
    def __init__(self, decay = decay):
        self.decay = decay
        pass

    def forward(
        self, y_true, y_pred, w_2, float = W_2
    ):
        self.y_pred = y_pred
        self.y_true = y_true

        self.loss = (
            -(np.log(self.y_pred + 1e-5) * self.y_true).sum() + 0.5 * self.decay * w_2
        )

        return self.loss
    
    def backward(self):
        return -(
            self.y_true / self.y_pred
        ) # [batchsize, n_out] / [batchsize, n_out] -> [batchsize, n_out]




def main():

    na_train_X = np.load('/Users/qjm/Documents/Usyd/2021/DL/Assignment 1/Assignment1-Dataset/train_data.npy')
    na_train_y = np.load('/Users/qjm/Documents/Usyd/2021/DL/Assignment 1/Assignment1-Dataset/train_label.npy')
    
    na_test_X = np.load('/Users/qjm/Documents/Usyd/2021/DL/Assignment 1/Assignment1-Dataset/test_data.npy')
    na_test_y = np.load('/Users/qjm/Documents/Usyd/2021/DL/Assignment 1/Assignment1-Dataset/test_label.npy')
    

    one_hot_train_y = np.eye(len(np.unique(na_train_y)))[na_train_y.squeeze()]
    one_host_test_y = np.eye(len(np.unique(na_test_y)))[na_test_y.squeeze()]

    W = None
    b = None
# EPOCH = 500
# n_batchsize = 5
# dropout_ratio = 0.2
# learning_rate = 0.001
# momentum = 0.8
# decay = 0.01
# eps = 1e-5
# W_2 = 0



    layers = []

    # first hidden layer
    #0 - linear layer
    layers.append(
        Layer(128, 256, W, b, learning_rate, momentum, decay),
    )
    #1 - batc normalisation
    layers.append(BatchNormalisation(256, learning_rate, momentum, decay))
    #2 - ReLU activation
    layers.append(ReLU())
    #3 - Dropout
    layers.append(Dropout())

    # output layer
    #4 - linear layer
    layers.append(
        Layer(256, 10, W, b, learning_rate, momentum, decay),
    )

    #5 - batch normalisation
    layers.append(BatchNormalisation(10, learning_rate, momentum, decay))
    #6 - softmax layer
    layers.append(Softmax())
    #7 - cross entropy
    layers.append(CrossEntropyLoss())



    EPOCH = 10
    for e in range(EPOCH):
        for i in range(0, len(na_train_X) - n_batchsize, n_batchsize):
            
            # -------------------------start: 1st hidden layer-------------------------
            forward_output = layers[0].forward(na_train_X[i:i+5])

            # print(forward_output.shape) # correct -> (5,256)
            
            # checked that the shape before and after batch normaliastion is the same
            normalised_output = layers[1].forward(forward_output)


            print('Batch normalisation should return n_out, n_out but its returning batchsize n_out')
            # print(normalised_output.shape) #5,256

            print('*****************')

            activation_output = layers[2].forward(normalised_output)

            dropout_output = layers[3].forward(activation_output)
            

            # -------------------------end: 1st hidden layer-------------------------
            


            # +++++++++++++++++++++++++start: last output layer+++++++++++++++++++++++++

            # print(dropout_output.shape) # shape (5,256)

            # output layer
            outputLayer_hidden = layers[4].forward(dropout_output)
            # print(outputLayer_hidden.shape) #(5,10)
            
            outputLayer_batchNorm = layers[5].forward(outputLayer_hidden)

            # print(outputLayer_batchNorm.shape) (5,10)
            
            outputLayer_Softmax = layers[6].forward(outputLayer_batchNorm)     
            # outputLayer_Softmax shape (5,10)

            outputLayer_crossEntropy = layers[7].forward(outputLayer_Softmax,
                                                        na_train_y[i:i+n_batchsize],
                                                        W_2
                                )
            
            print(outputLayer_crossEntropy) # one value of loss
            # +++++++++++++++++++++++++end: last output layer+++++++++++++++++++++++++



            # *************************start: output layer*************************
   

            # \\\\\\\\\\\\\\\\\\\\\\\\start : backward started with crossEntropy\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
            delta = layers[-1].backward()

            for layer in reversed(layers[:-1]):
                
                # iteration starts from Softmax

                if delta is not None:
                    delta = layer.backward(delta)
                    print('++++')
                    print(delta.shape)
                    print('----')
                else:
                    print(str(layer) +  ' found delta with None value')
                    exit()
            
            # \\\\\\\\\\\\\\\\\\\\\\\\\end : backward started with crossEntropy\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
            
            # Final: update
            layers[0].update(delta)

            # update

    exit()                
    




if __name__ == "__main__":
    main()





