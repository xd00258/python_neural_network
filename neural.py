from numpy import exp, array, random, dot

class NeuronLayer():
    #initialize weights with random values
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1

class NeuralNetwork():
    #add layers to network
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    #sigmoid function
    def __sigmoid(self, x):
        return 1/(1+exp(-x))

    def __sigmoid_derivative(self,x):
        return x*(1-x)

    #training function
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):

            #pass training set through neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            #calculate error for layer 2 (desired output minus predicted output)
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            #calculate error for layer 1 (with layer 2 error)
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            #calculate weight adjustments
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            #update weights
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    #thinking function
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2
    
    #printing function
    def print_weights(self):
        print (" Layer 1 (4 neurons, 2 inputs each): ")
        print (self.layer1.synaptic_weights)
        print (" Layer 2 (1 neuron, 4 inputs each): ")
        print (self.layer2.synaptic_weights)

if __name__ == "__main__":

    #fix a seed for random numbers (for practice)
    random.seed(1)

    #Layer 1
    layer1 = NeuronLayer(4,2)

    #Layer 2
    layer2 = NeuronLayer(1,4)

    #adding layers to neural network
    neural_network = NeuralNetwork(layer1, layer2)

    print ("Step 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    #training set
    training_set_inputs = array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
     ])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    #train it
    neural_network.train(training_set_inputs, training_set_outputs, 50000)

    print ("Stage 2) New synaptic weigths after training: ")
    neural_network.print_weights()

    #test with new situation
    new_situation = array([0,0])
    hidden_state, output = neural_network.think(new_situation)
    print ("Stage 3) Doing new situation: ", new_situation)
    print (output)