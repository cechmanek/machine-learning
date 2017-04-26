# k-nearest neighbors classifier

import numpy as np
import operator
from os import listdir

def createDataSet():
	group - array([[1.0,1.1],[1.0,1.0],[0.0,0.0],[0.0,0.1]])
	labels = ['A','B','C','D']
	return group, labels

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndices = distances.argsort()

	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndices[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

	return sortedClassCount[0][0]

def file2matrix(filename):
	fileHandle = open(filename)
	numerOfLines = len(fileHandle.readLines())
	returnMatrix = zeros(numberOfLines,3)
	classLabelVector = []
	fileHandle = open(filename) # reopen, as readLines moves us to the bottom

	index = 0
	for line in fileHandle.readLines():
		line = line.strip()
		listFromLine = line.split('\t')
		returnMatrix[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1

	return returnMatrix, classLabelVector

def autoNorm(dataSet):
	minVals = dataSet.min()
	maxVals = dataSet.max()
	ranges = maxVals - minVals

	normDataSet = np.zeros(np.shape(dataSet))
	m = dataSet.shape[0]

	normDataSet = dataSet - np.tile(minVals, (m,1))
	normDataSet = normDataSet/np.tile(ranges, (m,1))

	return normDataSet, ranges, minVals

def datingClassTest():
	hoRatio = 0.1

	datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)

	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)

	errorCount = 0.0

	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
		print("the classifer came back with %d, the real answer is: %d" % (classifierResult, datingLabels[i]))

		if (classifierResult != datingLabels[i]):
			errorCount += 1.0

	print("the total error rate is: %d" % errorCount/float(numTestVecs))

def classifyPerson():
	resultList = ['not at all', 'in small doses', 'in large doses']
	
	percentTats = float(raw_input("percenta of time spent playing video games?"))
	ffMiles = float(raw_input("frequent flier miles earned per year?"))
	iceCream = float(raw_input("liters of ice cream consumed per year?"))

	datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = np.array([ffMiles, percentTats, iceCream])

	classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)

	print("you will probably like this person: ", resultList[classifierResult - 1])

def img2vector(fileName):
	returnVect = np.zeros((1,1024))
	fr = open(fileName)

	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])

	return returnVect	

def handWritingClassTest():
	hwLabels = []

	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = np.zeros((m,1024))

	# build full model
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)

	# now that model is built, use it to test each example
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)

		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print("the classifier cam back with: %d, the real answer is %d" % (classifierResult, classNumStr))

		if (classifierResult != classNumStr) :
			errorCount += 1.0

	print("the total number of errors is: %d" % errorCount)
	print("the total error rate is: %d" % (errorCount/float(mTest)))