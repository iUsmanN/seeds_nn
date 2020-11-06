# Created by Usman Nazir on 05-11-2020
from random import random
from math import exp

##################################### NETWORK CREATION

def initialize_Network(n_inputs, n_hidden, n_outputs):
    #List of layers
    network = list();

    #A layer with n_hidden neurons and each weight having a n_inputs + 1(bias) random weights.
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]

    #append the hidden layer
    network.append(hidden_layer)

    # A layer with n_outputs neurons and each weight having a n_hidden + 1(bias) random weights.
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]

    #append output layer
    network.append(output_layer)

    #return the network
    return network



##################################### FORWARD PROPAGATION

def z(inputs, weights):
    #Initially sets the z value to the last value of the weights (bias)
    z_value = weights[-1]
    for i in range(len(weights)-1):
        z_value = z_value + (inputs[i] * weights[i])
    return z_value

def g(z):
    return 1 / (1 + exp(-z))

#Forward propagation
def forwardPropagate(network, row):

    # Set inputs
    inputs = row

    #Runs for both the hidden and the output layer
    for layer in network:

        #Holds new inputs
        new_inputs = []

        #runs for each neuron on the layer
        for neuron in layer:

            #calculates the z value
            z_value = z(inputs, neuron['weights'])

            #Calculates the g value
            g_value = g(z_value)

            #Sets the neuron's output_value
            neuron['output'] = g_value

            #Adds the output value to the new inputs for next layer
            new_inputs.append(neuron['output'])

        #Sets the previous layers output to the inputs of the next layer
        inputs = new_inputs

    #Returns the final output values
    return inputs



##################################### BACKWARD PROPAGATION

def g_derivative(x):
    return x * (1.0 - x)

def back_propagation(network, expected_values):

    #Go from the last layer till the first one
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()

        # if this is not the final layer
        if i != len(network)-1:

            # iterate through all the neurons in the layer
            for j in range(len(layer)):
                error = 0.0

                # for all neurons in its next layer
                for neuron in network[i + 1]:

                    # calculate the error using the this neuron in the next layer
                    error += neuron['weights'][j] * neuron['delta']
                errors.append(error)

        # if this is the final layer
        else:

            # for all the output neurons
            for j in range(len(layer)):

                #output neuron(j)
                neuron = layer[j]

                #calculate the error value
                errors.append(expected_values[j] - neuron['output'])

        # for all neurons in whatever layer
        for j in range(len(layer)):

            #neuron on the layer
            neuron = layer[j]

            #calculate delta value which is used in hidden layer calculations on the previous layer
            neuron['delta'] = errors[j] * g_derivative(neuron['output'])


def update_weights(network, row, learning_rate):

    #for each layers in the network
    for i in range(len(network)):

        #get all columns for the input except the last one
        inputs = row[:-1]

        #if this is not the first layer
        if i != 0:

            #input would be the output of the prevvious layer
            inputs = [neuron['output'] for neuron in network[i-1]]

        #for all neurons in this layer
        for neuron in network[i]:

            #for each input value j
            for j in range(len(inputs)):

                #update the weight j
                neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]

            #update the bias weights
            neuron['weights'][-1] += learning_rate * neuron['delta'] * 1




##################################### TRAINING

def train_network(network, training_data, learning_rate, n_epoch, n_outputs):

    #For each pass in the training data
    for epoch in range(n_epoch):

        #Initially error is 0
        sum_error = 0

        #for each row in the learning data
        for row in training_data:

            #Get values/outputs for each neuron after forward propagation
            outputs = forwardPropagate(network, row)

            #Expected output array
            expected = [0 for i in range(n_outputs)]

            #?
            expected[row[-1]] = 1

            #Calculate the sum of errors at the end
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])

            #Back propagate the errors
            back_propagation(network, expected)

            #update the weights
            update_weights(network, row, learning_rate)

        #Print the current values
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))



##################################### PREDICTION

def predict(network, row):

    #Calculate the outputs
    outputs = forwardPropagate(network, row)
    return outputs.index(max(outputs))


##################################### RUNNING

dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

#Get number of inputs
n_inputs = len(dataset[0]) - 1

#Get number of outputs
n_outputs = len(set([row[-1] for row in dataset]))

#Initialize NN with 2 hidden layers
network = initialize_Network(n_inputs, 2, n_outputs)

#Train NN
train_network(network, dataset, 0.5, 20, n_outputs)

#Check Predictions
for row in dataset:
	prediction = predict(network, row)
	print('Expected=%d, Got=%d' % (row[-1], prediction))