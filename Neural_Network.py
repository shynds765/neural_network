import numpy as np

class Neural_Network():
    def __init__(self,layer_sizes):
        self.layer_sizes = layer_sizes
        self.layers = []

        for i in range(len(layer_sizes)-1):
            self.layers.append(self.Neuron_Layer(layer_sizes[i],layer_sizes[i+1]))
            self.layers.append(self.Activation_Layer())

    def train(self,X,Y,iterations):
        for i in range(iterations):
            error = 0
            for x,y in zip(X,Y):
                y_pred = self.predict(x)
                error += self.MSE(y,y_pred)
                gradient = self.MSE_prime(y,y_pred)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient,.2)
    
    def predict(self,input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    class Layer:
        def __init__(self):
            self.input = None
            self.output = None    
        def forward(self,input):
            pass
        def backward(self,gradient_out,step_size):
            pass
    class Neuron_Layer(Layer):
        def __init__(self,input_size,output_size):
            self.weights = np.random.randn(output_size,input_size) #Randomly initialize weights
            self.biases = np.random.randn(output_size,1) #Randomly initalize biases
        
        def forward(self,input):
            self.input = input
            return np.matmul(self.weights,input) + self.biases

        def backward(self, gradient_out, step_size):
            gradient_weights = np.matmul(gradient_out,self.input.T)
            self.weights -= step_size*gradient_weights
            self.biases -= step_size*gradient_out #output gradient is equal to biases gradient
            return np.matmul(self.weights.T,gradient_out)

    class Activation_Layer(Layer):
        def __init__(self):
            pass

        def forward(self,input):
            self.input = input
            return self.sigmoid(input)
        
        def backward(self,gradient_out,step_size):
            return np.multiply(gradient_out,self.sigmoid(self.input)*(1-self.sigmoid(self.input)))

        def sigmoid(self,input):
            return 1/(1+np.exp(-input))

    def MSE(self,y,y_pred):
        return np.mean(np.power(y-y_pred, 2))

    def MSE_prime(self,y,y_pred):
        return 2*(y_pred-y)/np.size(y)

nn = Neural_Network([2,100,2])
x = np.array([[[1],[0]],[[0],[1]],[[0],[0]],[[1],[1]]])
y = np.array([[[1],[0]],[[1],[0]],[[0],[1]],[[0],[1]]])

nn.train(x,y,10000)

testData = np.array([[[0],[1]],[[0],[0]],[[0],[1]],[[1],[0]],[[1],[1]],[[1],[1]],[[0],[1]]])
for i in testData:
    output = nn.predict(i)
    if output[0][0] > output[1][0]:
        print("True")
    else:
        print("False")