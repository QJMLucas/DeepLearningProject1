import numpy as np
from HiddenLayer import HiddenLayer
from Activation import Activation
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import math


from scipy.special import softmax

class MLP:
    """
    """ 

    # for initiallization, the code will create all layers automatically based on the provided parameters.     
    def __init__(self, layers, activation=[None,'relu','softmax']):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """        
        ### initialize layers
        self.layers=[]
        self.params=[]

        self.totalCount = 0
        self.successCount = 0

        self.stopCounter = 0
        
        self.activation=activation
        for i in range(len(layers)-1):
            self.layers.append(HiddenLayer(layers[i],layers[i+1],activation[i],activation[i+1]))


    # forward progress: pass the information through the layers and out the results of final output layer
    def forward(self,input):
        for layer in self.layers:
            output=layer.forward(input)
            input=output
            
        return input

    # define the objection/loss function, we use mean sqaure error (MSE) as the loss
    # you can try other loss, such as cross entropy.
    def crossEntrophy(self,y,y_hat):
        activation_deriv=Activation(self.activation[-1]).f_deriv
        # MSE

        # print(y_hat)
        # exit()
        # y represents class label - starts from 0 to 9
        # loss,delta=self.crossEntrophy(y[i],y_hat) - y[i] is one row of data with 128 columsn and y_hat is the result of the forward pro.

        # error example: [-1.42078735]
        # error needs to be the derivative of cross entrophy which is not y minus y_hat

        # error = y-y_hat

        y_hat[y_hat==0] = 1e-9
    

        # loss=error**2

        # 根据 dataset 里的class
      

        Y_true_label = np.zeros(len(y_hat))

        # row 1 - 5 , 

        # Below is correct y - 0-9
        Y_true_label[y] = 1

        Y_true= 1
        
        print(Y_true_label)

        print(y_hat)
    

        # crossEntrop - my version - correct loss
        loss = -np.sum(Y_true_label * np.log(y_hat))
        # loss = -(Y_true_label* np.log(y_hat))

        # error should be the derivative of cross entrophy which is below:
        # np.log(e) * 1/x - since np.log(e) is = 1 so:
        # error = 1/y_hat
   
        # print(loss)

        # get the largest prob
        largest_prob = np.amax(y_hat)
        # find the index
        largest_prob_index = np.where(y_hat == largest_prob)

        # print(y_hat)
        # print(largest_prob)
        # print(largest_prob_index)
        # print(type(largest_prob_index))

        list_largest_pro_index = []
        for x in largest_prob_index:
            if len(list_largest_pro_index) <1:
                list_largest_pro_index.append(x)

         
        if len(list_largest_pro_index[0]) == 2:
            if int(Y_true_label[list_largest_pro_index[0][0]]) == 1 or int(Y_true_label[list_largest_pro_index[0][1]]) == 1:
                temp = 1
            else:
                temp = 0
        elif len(list_largest_pro_index[0]) == 1:
            if int(Y_true_label[list_largest_pro_index[0][0]]) == 1:
                temp = 1
            else:
                temp = 0
        else:
            temp = 0
    
            
        # calculate the delta of the output layer
            
        # delta=-error*activation_deriv(y_hat)

        # delta = y_hat- y
        delta = 1/y_hat


        # return loss and delta
        return loss,delta,temp

    # backward progress  
    def backward(self,delta):
        delta=self.layers[-1].backward(delta,output_layer=True)
        for layer in reversed(self.layers[:-1]):
            delta=layer.backward(delta)

    # update the network weights after backward.
    # make sure you run the backward function before the update function!    
    def update(self,lr):
        
        # gamma term is usally set to 0.9 as mentioned by lecturer
        momentum_gamma = 0.98

        for layer in self.layers:
            
            # momentum
            layer.weight_V_t = momentum_gamma * layer.weight_V_t + lr * layer.grad_W
            layer.W =  layer.W - layer.weight_V_t
            # layer.W = (1-lr*0.98)*layer.W - layer.weight_V_t

            # weight decay - extra update to shrink the weight
            layer.W *= 0.98

            layer.bias_V_t = momentum_gamma * layer.bias_V_t + lr * layer.grad_b
            layer.b = layer.b - layer.bias_V_t


            # one sample Stochastic gradient descent 
            # layer.W -= lr * layer.grad_W
            # layer.b -= lr * layer.grad_b

    
    # def cross_entropy(self, actual, predicted):
    #     loss = -np.sum(actual * np.log(predicted))
    #     return loss

    # define the training function
    # it will return all losses within the whole training process.
    def fit(self,X,y,learning_rate=1e-4, epochs=100):
        """
        Online learning.
        :param X: Input data or features
        :param y: Input targets
        :param learning_rate: parameters defining the speed of learning
        :param epochs: number of times the dataset is presented to the network for learning
        """ 
        X=np.array(X)
        y=np.array(y)
        to_return = np.zeros(epochs)
        
        for k in range(epochs):
            loss=np.zeros(X.shape[0])
            for it in range(X.shape[0]):

                i=np.random.randint(X.shape[0])

                print('Step 1 forward -start')



                # forward pass
                y_hat = self.forward(X[i])

                print(y_hat)
                print(sum(y_hat))

                print('89757')



                self.stopCounter+=1
                # if self.stopCounter == 2:
                #     exit()
        

                # print(y[i])
                # print('one iteration of the forward finished here')


                print('Step 2 - loss start')

                # backward pass
             
                loss,delta,successCount=self.crossEntrophy(y[i],y_hat)

                # print(loss)
                # print('loss above')

                self.backward(delta)
                
                y
                self.update(learning_rate)
                self.totalCount+=1
                self.successCount+=successCount
            
            print(self.totalCount)
            print(self.successCount)

            to_return[k] = np.mean(loss)
            total_val = self.totalCount
            total_succ = self.successCount
            

        return to_return, total_val, total_succ

    # define the prediction function
    # we can use predict function to predict the results of new data, by using the well-trained network.
    def predict(self, x):
        x = np.array(x)
        output = np.zeros(x.shape[0])
        for i in np.arange(x.shape[0]):
            output[i] = self.forward(x[i,:])
        return output

def main():
    
    # x = np.array([709.79162097,709.79162097,709.79162097,709.79162097,709.79162097, 709.79162097,709.79162097,709.79162097,709.79162097,709.79162097])
    # np.exp(x)/np.sum(np.exp(x))

    input_data = np.load('/Users/qjm/Documents/Usyd/2021/DL/Assignment 1/Assignment1-Dataset/test_data.npy')
    output_data = np.load('/Users/qjm/Documents/Usyd/2021/DL/Assignment 1/Assignment1-Dataset/test_label.npy')
    


    # print(np.unique(output_data))
    # print(output_data[7])
    # value + 1  - index 7 is value 6

    # Lucas note: there is 10 different classes from 0 to 9 so 10 classes in total.
    # print(np.unique(output_data))

    # 10 classes determines the output of the forward layer

    layers = [128,256, 10]
    activation = [None,'relu', 'softmax']
    nn = MLP(layers, activation)
    aa,total,total_success = nn.fit(input_data, output_data, learning_rate=0.001, epochs=1)
    print('loss:%f'%aa[-1])
    print(total)
    print(total_success)

    print(total_success/total)

    # 128 is the input shape, second value is the number of neurons in the relu layer and 1 neuron in the tanh output layer
    # 1 neuron in the tanh output layer should be changed to the number of classes you want to predict

if __name__ == "__main__":
    main()

# learning rate
# activation function
# neuro
# mini-batch
