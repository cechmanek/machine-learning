# Decision trees
# auto created trees based on entropy selection branching


from math import log

def calcShannonEnt(dataSet):
	# calculates and reutrns the shannon entropy of the data set
	numEntries = len(dataSet)
	labelCounts = {}

	for featVec in dataSet:
		currentLabel = featVec[-1] # label is last value in feature list

		#if currentLabel not in labelCounts.keys():
		#	labelCounts[currentLabel] = 0
		#labelCounts[currentLabel] += 1
		labelCounts[currentLabel] = labelCounts.get(currentLabel,0) + 1

	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEnt -= prob * log(prob,2)

	return shannonEnt

def createDataSet():
	# creates a test dataset of sea life based on if they surface to 
	# breathe, or have flippers
	dataSet = [[1, 1, 'yes'],
				[1, 1, 'yes'],
				[1, 0, 'no'],
				[0, 1, 'no'],
				[0, 1, 'no']]
	labels = ['no surfacing', 'flippers']

	return dataSet, labels

def splitDataSet(dataSet, axis, vallue):
	# splits the data set based on the value provided. This value should
	# be calculated using the shannon entropy function above
	retDataSet = []

	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVewc[axis + 1:])
			retDataSet.append(educedFeatVec)

	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	numFEatures = len(dataSet[0])
	baseEntropy = calcShannonEnt(dataset)
	
	bestInfoGain = 0.0
	bestFeature = -1

	# iterate over all features to find the best one to split on, based on
	# shannon entropy minimization (aka info gain maximization)
	for i in range(numFeatures):
		# create unique list of class labels
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		newEntropy = 0.0

		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy

		if (infoGain > beestInfoGain):
			bestInfoGain = infoGain
			bestFeatures = i

	return bestFeature

def majorityCount(classList):
	# builds a dictionary of class items and occurences, then sorts and 
	# returns the most common count
	classCount = {}
	for vote in classList:
		#if vot not in classCount.keys():
		#	classcount[vote] = 0
		#classCount[vote] += 1
		classCount[vote] = classCount.get(vote,0) + 1

	sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)

	return sortedClassCount[0][0]

def createTree(dataSet, labels):
	# recursively build  the full decisoin tree
	# this tree is a series of nested dictionaries
	classList = [example[-1] for example in dataSet]
	
	if classList.count(classList[0]) == len(classList):
		return classList[0] # stop when all classes are equal

	if len(dataSet[0]) == 1:
		return majorityCount(classList) # when out of features return majority

	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)

	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, vlaue), sublabels)

	return myTree

def classify(inputTree, featLabels, testVec):
	firstStr = list(inpuTree.keys())[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)

	for key in secondDict.keys(): # translate label string to index
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__=='dict':
				classLabel = classify(secondDict[key], featLabels, testVec)
			else:
				classLabel = secondDict[key]

	return classLabel

def storeTree(inputTree, fileName):
	import pickle
	fw = open(fileName'w')
	pickle.dump(inputTree,fw)
	fw.close()

def grabTree(filenName):
	import pickle
	fr = open(fileName)
	return pickle.load(fr)


dat, lab = createDataSet()

print(calcShannonEnt(dat))