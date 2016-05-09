import Mariana.activations as MA
import Mariana.decorators as MD
import Mariana.layers as ML
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS

import Mariana.settings as MSET

from Mariana.examples.useful import load_mnist

import numpy as np
import cPickle

MSET.VERBOSE = False

if __name__ == "__main__" :


	#make fake data
	numEx = 2000
	numFeat = 300
	
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
		
	train_set = make_train_set(numEx, numFeat, 0.0001)
	
	validation_set = make_train_set(numEx, numFeat, 0.0001)
	
	#Define cost and learning scenario
	ls = MS.GradientDescent(lr = 0.01)
	cost = MC.NegativeLogLikelihood()

	#Define layers here
	
	inp = ML.Input(numFeat, name = "InputLayer")

	hid1 = ML.Hidden(100,
		activation = MA.ReLU(),
		regularizations = [MR.L2(0.0001)],
		decorators = [MD.BinomialDropout(0.1)],
		name = "Hidden1"
	)
	
	hid2 = ML.Hidden(10,
		activation = MA.ReLU(),
		regularizations = [MR.L2(0.0001)],
		decorators = [MD.BinomialDropout(0.1)],
		name = "Hidden2"
	)

	out = ML.SoftmaxClassifier(2,
		learningScenario = ls,
		costObject = cost,
		regularizations = [MR.L2(0.0001)],
		name = "OutputLayer"
	)

	#Define network here
	#MLP = inp > hid1 > hid2 > out
	MLP = inp > out
	
	#save html
	MLP.saveHTML("Monbug_mlp")


	trainScores = []
	miniBatchSize = 10
	for epoch in xrange(10) :
		print 'epoch: ', epoch
		for i in xrange(0, len(train_set[0]), miniBatchSize) :
			inputs = train_set[0][i : i +miniBatchSize]
			targets = train_set[1][i : i +miniBatchSize]
			score = MLP.train("OutputLayer", InputLayer = inputs, targets = targets)
			trainScores.append(score)

		trainScore = np.mean(trainScores)
		
		valScore = MLP.test("OutputLayer", InputLayer = validation_set[0], targets = validation_set[1])[0]
		print "---\nepoch: %s. train: %s, validation: %s" %(epoch, trainScore, valScore)


# ReLU, GradientDescent, NegativeLogLikelihood, BinomialDropout, SoftmaxClassifier, learningScenario, costObject, Input, InputLayer, Hidden, OutputLayerm saveHTML
