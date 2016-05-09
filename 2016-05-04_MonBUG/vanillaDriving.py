import Mariana.activations as MA
import Mariana.decorators as MD
import Mariana.layers as ML
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS
import numpy as np

ls = MS.GradientDescent(lr = 0.01)
cost = MC.NegativeLogLikelihood()

i = ML.Input(28*28, name = "inputLayer")
h = ML.Hidden(300, activation = MA.ReLU(), decorators = [MD.BinomialDropout(0.2)], regularizations = [ MR.L1(0.0001) ])
o = ML.SoftmaxClassifier(2, learningScenario = ls, costObject = cost, regularizations = [ MR.L1(0.0001) ])

MLP = i > h > o

	# make data
def make_train_set(numEx, numFeat, noise=0.01):
	trainNoise = noise
	labels = np.random.choice([0,1], size=numEx)
	values = np.zeros((numEx,numFeat)) + np.random.rand(numEx,numFeat)*trainNoise
	for i in range(len(labels)):
		values[i] = abs(labels[i] - values[i])
	train_set = [
				values.tolist(),
				labels.tolist()
				]
	return train_set

numEx = 10000
numFeat = 28*28
train_set = make_train_set(numEx, numFeat)
test_set = make_train_set(numEx, numFeat)

miniBatchSize = 100
j=0
#train the model for output 'o' function will update parameters and return the current cost
print MLP.train(o, inputLayer = train_set[0][j : j +miniBatchSize], targets = train_set[1][j : j +miniBatchSize] )

#the same as train but does not updated the parameters
print MLP.test(o, inputLayer = test_set[0][j : j +miniBatchSize], targets = test_set[1][j : j +miniBatchSize] )
print test_set[1][j : j +miniBatchSize]
#the propagate will return the output for the output layer 'o'
print MLP.propagate(o, inputLayer = test_set[0][j : j +miniBatchSize])