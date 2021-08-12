import numpy as np
HIDDEN_LAYER_SIZE = 4

def sigmoid(x): 
    return 1/(1+np.exp(-x)) 

def sigmoid_derivative(p):
    return p*(1-p)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.y          = y
        # self.layers     = np.
        self.weights1   = np.random.rand(self.input.shape[1],HIDDEN_LAYER_SIZE) #returns a random matrix that is (# input nodes)x(# nodes in hidden layer)
        self.weights2   = np.random.rand(HIDDEN_LAYER_SIZE,1)                   #returns a random matrix that is (# nodes in hidden layer)x(# nodes in output)
        # self.biases1    = np.random.rand(1, HIDDEN_LAYER_SIZE)
        # self.biases2    = np.random.rand(1, self.y.shape[1])
        self.output     = np.zeros(self.y.shape[1])                #initializes an array with 0 for the output values

    def feedforward(self):
        # self.layer1 = sigmoid(np.dot(self.input, self.weights1) + self.biases1)
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))

        # self.output = sigmoid(np.dot(self.layer1, self.weights2) + self.biases2)
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return self.output


    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        # d_biases2 = 2*(self.y - self.output) * sigmoid_derivative(self.output)
        
        self.weights2 += d_weights2
        # self.biases2 += d_biases2

        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))       
        # d_biases1 = 2*(self.y - self.output) * sigmoid_derivative(self.output) * sigmoid_derivative(self.layer1)

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        # self.biases1 += d_biases1
        

    def train(self, x, y):
        self.output = self.feedforward()
        self.backprop()