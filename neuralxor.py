import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_d(z):
    return z*(1-z)

class XOR_NEURAL_NETWORK:
    def __init__(self):
        self.X = np.array([[0,0],[1,0],[0,1],[1,1]])
        self.Y = np.array([[0],[1],[1],[0]])
        
        self.hidden_weights = np.random.rand(2,2)
        self.hidden_biases = np.random.rand(1,2)
        self.output_weights = np.random.rand(2,1)
        self.output_biases = np.random.rand(1,1)

        self.lr = 0.1
        self.epoches = 15000

    def __prop_forward(self):
        self.hidden_layer = sigmoid(np.dot(self.X, self.hidden_weights) + self.hidden_biases)
        self.prediction = sigmoid(np.dot(self.hidden_layer, self.output_weights) + self.output_biases)

    def __prop_back(self):
        self.prediction_d = (self.Y-self.prediction) * sigmoid_d(self.prediction)
        self.hidden_layer_d = self.prediction_d.dot(self.output_weights.T) * sigmoid_d(self.hidden_layer)
    
    def __update(self):
        self.output_weights += self.hidden_layer.T.dot(self.prediction_d) * self.lr
        self.output_biases += np.sum(self.prediction_d) * self.lr
        self.hidden_weights += self.X.T.dot(self.hidden_layer_d) * self.lr
        self.hidden_biases += np.sum(self.hidden_layer_d) * self.lr

    def train(self):
        for i in range(self.epoches):
            self.__prop_forward()
            self.__prop_back()
            self.__update()

    def print_prediction(self):
        for i in range(4):
            print(self.X[i],"-->",self.prediction[i],"-->",self.Y[i])

    
def main():
    xorn = XOR_NEURAL_NETWORK()
    xorn.train()
    xorn.print_prediction()


if __name__ == "__main__":
    main()
