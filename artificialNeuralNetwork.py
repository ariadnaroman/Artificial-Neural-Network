from random import *
from math import *
import csv


class Neuron:
    def __init__(self, w=[], out=None, delta=0.0):
        self.weights = w
        self.output = out
        self.delta = delta

    def __str__(self):
        return "weights: " + str(self.weights) + ", output: " + str(self.output) + ", delta: " + str(self.delta)

    def __repr__(self):
        return "weights: " + str(self.weights) + ", output: " + str(self.output) + ", delta: " + str(self.delta)


def netInitialisation(noInputs, noOutputs, noHiddenNeurons):
    net = []
    hiddenLayer = [Neuron([random() for i in range(noInputs + 1)]) for h in range(noHiddenNeurons)]
    net.append(hiddenLayer)
    hiddenLayer2 = [Neuron([random() for i in range(noInputs + 1)]) for h in range(noHiddenNeurons)]
    net.append(hiddenLayer2)
    outputLayer = [Neuron([random() for i in range(noHiddenNeurons + 1)]) for o in range(noOutputs)]
    net.append(outputLayer)
    return net


def activate(input, weights):
    result = 0.0
    for i in range(0, len(input)):
        result += float(input[i]) * float(weights[i])
    result += weights[len(input)]
    return result


def transfer(value):
    return 1.0 / (1.0 + exp(-value))


def forwardPropagation(net, inputs):
    for layer in net:
        newInputs = []
        for neuron in layer:
            activation = activate(inputs, neuron.weights)
            neuron.output = transfer(activation)
            newInputs.append(neuron.output)
        inputs = newInputs
    return inputs


def transferInverse(val):
    return val * (1 - val)


def backwardPropagation(net, expected):
    for i in range(len(net) - 1, 0, -1):
        crtLayer = net[i]
        errors = []
        if i == len(net) - 1:  # last layer
            for j in range(0, len(crtLayer)):
                crtNeuron = crtLayer[j]
                errors.append(expected[j] - crtNeuron.output)
        else:  # hidden layers
            for j in range(0, len(crtLayer)):
                crtError = 0.0
                nextLayer = net[i + 1]
                for neuron in nextLayer:
                    crtError += neuron.weights[j] * neuron.delta
                errors.append(crtError)
        for j in range(0, len(crtLayer)):
            crtLayer[j].delta = errors[j] * transferInverse(crtLayer[j].output)


def updateWeights(net, example, learningRate):
    for i in range(0, len(net)):
        inputs = example[:-1]
        if i > 0:
            inputs = [neuron.output for neuron in net[i - 1]]
        for neuron in net[i]:
            for j in range(len(inputs)):
                neuron.weights[j] += learningRate * neuron.delta * inputs[j]
        neuron.weights[-1] += learningRate * neuron.delta


def training(net, data, noOutputTypes, learningRate, noEpochs):
    for epoch in range(0, noEpochs):
        sumError = 0.0
        for example in data:
            inputs = example[:- 1]
            computedOutputs = forwardPropagation(net, inputs)

            expected = [0 for i in range(noOutputTypes)]
            expected[example[-1]] = 1
            computedLabels = [0 for i in range(noOutputTypes)]
            computedLabels[computedOutputs.index(max(computedOutputs))] = 1
            computedOutputs = computedLabels

            crtErr = sum([(expected[i] - computedOutputs[i]) ** 2 for i in range(0, len(expected))])

            sumError += crtErr
            backwardPropagation(net, expected)
            updateWeights(net, example, learningRate)


def evaluating(net, data, noOutputTypes):
    computedOutputs = []
    for inputs in data:
        computedOutput = forwardPropagation(net, inputs[:-1])

        computedLabels = [0 for i in range(noOutputTypes)]
        computedLabels[computedOutput.index(max(computedOutput))] = 1
        computedOutput = computedLabels

        computedOutputs.append(computedOutput[0])
    return computedOutputs


def computePerformance(computedOutputs, realOutputs):
    noOfMatches = sum([computedOutputs[i] == realOutputs[i] for i in range(0, len(computedOutputs))])
    return noOfMatches / len(computedOutputs)


def run(trainData, testData, learningRate, noEpochs):
    test = []
    train = []
    for line in testData:
        testline = []
        for string in line[:-1]:
            string = float(string)
            testline.append(string)
        testline.append(line[-1])
        test.append(testline)
    for line in trainData:
        trainline = []
        for string in line[:-1]:
            string = float(string)
            trainline.append(string)
        trainline.append(line[-1])
        train.append(trainline)
    noInputs = len(train[0]) - 1
    noOutputs = len(set([example[-1] for example in train]))
    net = netInitialisation(noInputs, noOutputs, 2)
    training(net, train, noOutputs, learningRate, noEpochs)

    realOutputs = [test[i][j] for j in range(len(test[0]) - 1, len(test[0])) for i in range(0, len(test))]
    computedOutputs = evaluating(net, test[:-1], noOutputs)
    print("Accuracy: ", computePerformance(computedOutputs, realOutputs))


