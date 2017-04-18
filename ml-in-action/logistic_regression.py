# logistic regression classifier
# utilizes gradient ascent for the weight optimizing function

import numpy as np

def loadDataSet():
	dataMat = []
	labelMat = []

	fr = open('testSet.txt')

	for line in fr.readlines():
		lineArray = line.strip().split()
		dataMat.append([1.0, float(lineArray[0]), float(lineArray[1])])
		labelMat.append(int(lineArray[2]))

	return dataMat, labelMat

def sigmoid(inX):
	return 1.0/(1 + np.exp(-inX))

# batch processing gardient ascent
def gradAscent(dataMatIn, classLabels):
	dataMatrix = np.mat(dataMatIn)
	labelMat = np.mat(classLabels).transpose()

	m,n = np.shape(dataMatrix)
	
	alpha = 0.001
	maxCycles = 500

	weights = np.ones((n,1))

	for k in range(maxCycles):
		h = sigmoid(dataMatrix * weights) # matrix multiplication
		error = labelMat - h
		weights = weights  + alpha * dataMatrix.transpose() * error # matrix math, so don't use +=

	return weights

def plotBestFit(wei): # plots the best fit line based on weights
	from matplotlib import pyplot as plt

	weights = wei.getA() # returns itself as an n-d array object
	dataMat, labelMat = loadDataSet()
	dataArray = np.array(dataMat)

	n = np.shape(dataArray)[0]

	xCord1 = []
	yCord1 = []
	xCord2 = []
	yCord2 = []

	for i in range(n):
		if int(labelMat[i]) == 1:
			xCord1.append(dataArray[i,1])
			yCord1.append(dataArray[i,2])
		else:
			xCord2.append(dataArray[i,1])
			yCord2.append(dataArray[i,2])
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xCord1, yCord1, s=30, c='red', marker='s')
	ax.scatter(xCord2, yCord2, s=30, c='green')

	x = np.arange(-3.0, 3.0, 0.1)
	y = (-weights[0] - weights[1] * x) / weights[2]

	ax.plot(x, y)
	plt.xlabel('X1')
	plt.ylabel('X1')

	plt.show()

# stochastic gradient ascent
def stocGradAscent0(dataMatrix, classLabels):
	m,n = np.shape(dataMatrix)
	alpha = 0.01 # factor to incrementally update weights
	weights = np.ones(n)

	for i in range(m):
		h = sigmoid(sum(dataMatrix[i] * weights))
		error = classLabels[i] - h
		weights = weights * alpha * error * dataMatrix[i]

	return weights

def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
	m,n = np.shape(dataMatrix)
	
	weights = np.ones(n)

	for j in range(numIter):
		dataIndex = range(m)

		for i in range(m):
			alpha = 4/(1.0 + j+i) +0.01 # reduce alpha as you go through more iterations via 'j'
			# reduce alpha by data point index 'i' so alpha isn't strictly decreasing
			# (downside os this adds more an effect of data set order, but can assume it's random)			
			# randomly choose which data to use for weight update calc
			randIndex = int(np.random.uniform(0, len(dataIndex)))			

			h = sigmoid(sum(dataMatrix[randIndex] * weights))
			error = classLabels[randIndex] - h

			weights = weights + alpha * error * dataMatrix[randIndex]

	return weights


# utilizing logistic regression on horse colic data set

def classifyVector(inX, weights):
	prob = sigmoid(sum(inX * weights))

	if prob > 0.5:
		return 1.0
	else:
		return 0.0

def colicTest():
	frTrain = open('horseColicTraining.txt')
	frTest = open('horseColicTest.txt')

	trainingSet = []
	trainingLabels = []
	
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArray = []

		for i in range(21):
			lineArray.append(float(currLine[i]))

		trainingSet.append(lineArray)
		trainingLabels.append(float(currLine[21]))
	
	trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
	errorCount = 0
	numTestVec = 0.0

	for line in frTest.readlines():
		numTestVec += 1.0
		currLine = line.strip().split('\t')

		lineArray = []

		for i in range(21):
			lineArray.append(float(currLine[i]))

		if int(classifyVector(np.array(lineArray), trainWeights)) != int(currLine[21]):
			# class label is col 21 in data set, hence comparing to currLine[21]
			errorCount += 1

	errorRate = float(errorCount)/numTestVec
	print('the error rate of this test is: %f' % errorRate)

	return errorRate
		
def multiTest():
	numTests = 10
	errorSum = 0.0

	for k in range(numTests):
		errorSum += colicTest()

	print('after %d iterations the average error rate is: %f' %(numTests, errorSum/float(numTests)))