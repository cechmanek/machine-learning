# naive Bayes classifier

import numpy as np

def loadDataSet():
	postingList = [['my', 'dog', 'has', 'flea', \
	'problems', 'help', 'please'],
	['maybe', 'not', 'take', 'him', \
	'to', 'dog', 'park', 'stupid'], \
	['my', 'dalmation', 'is', 'so', 'cute', \
	'I', 'love', 'him'],
	['stop', 'posting', 'stupid', 'worthless', 'garbage',],
	['mr', 'licks', 'ate', 'my', 'steak', 'how', \
	'to', 'stop', 'him'],
	['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

	classVec = [0, 1, 0, 1, 0, 1] # 1 is abusive, 0 is not

	return postingList, classVec

def createVocabList(dataSet):
	vocabSet = set([])

	for document in dataSet:
		vocabSet = vocabSet | set(document) # cunion of sets

	return list(vocabSet)

'''
def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0]*len(vocabList)

	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print("the word: %s is not in my Vocabulary" % word)

	return returnVec
'''
# setOfWords2Vec was a binary yes/no counter of word presence.
# bagOfWords2Vect counts the number of occurances of a word
def bagOfWords2Vec(vocabList, inputSet):
	returnVec = [0] * len(vocabList)

	for word in inputSet:
		if word in vocabList:
			returnVect[vocabList.index(word)] += 1

	return returnVec

def trainNBO (trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])

	pAbusive = np.sum(trainCategory)/float(numTrainDocs)
	p0Num = np.ones(numWords) # using ones and 2.o, instead of zeros, 1 to avoid multiplying by 0
	p1Num = np.ones(numWords)
	p0Denom = 2.0
	p1Denom = 2.0

	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += np.sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += np.sum(trainMatrix[i])

	p1Vect = np.log(p1Num/p1Denom) # use log() to avoid underflow from small probabilities
	p0Vect = np.log(p0Num/p0Denom)

	return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1): # only 2 classes, pClass0=1-pClass1, so no need to pass it
	p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)
	p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0-pClass1)

	if p1 > p0:
		return 1
	else:
		return 0

def testingNB():
	listOfPosts, listClasses = loadDataSet()
	myVocabList = createVocabList(listOfPosts)
	trainMat = []

	for postInDoc in listOfPosts:
		#trainMat.append(setOfWords2Vec(myVocabList, postInDoc))
		trainMat.append(bagOfWords2Vec(myVocabList, postInDoc))

	p0V, p1V, pAb = trainNBO(np.array(trainMat), np.array(listClasses))

	# test with a good post
	testEntry = ['love', 'my', 'dalmation']
	#thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
	thisDoc = np.array(bagOfWords2Vec(myVocabList, testEntry))
	print(testEntry," classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))

	# test with an abusive post
	testEntry = ['stupid', 'garbage']
	#thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
	thisDoc = np.array(bagOfWords2Vec(myVocabList, testEntry))
	print(testEntry," classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))


# basic text parsing section
import re

def textParse(bigString):
	listOfTokens = re.split(r'\W*', bigString)
	return [token.lower() for token in listOfTokens if len(token) > 2]

def spamTest():
	docList = []
	classList = []
	fulltext = []

	# loop through and read the emails in the email folder
	for i in range(1,26):
		wordList = textParse(open('email/spam/%d.txt' % i).read())	
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1) # add class label as 1='spam'

		wordList = textParse(open('email/ham/%d.txt' % i,).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0) # add class labale as 0='not spam'

	vocabList = creatVocabList(docList)
	trainingSet = range(50) # 50 emails total
	testSet = []

	for i in range(10): # choose 10 of the 50 for the test set
		# choose a random example, add it to test set, remove from training set
		randIndex = int(np.random.uniform(0, len(traingingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])

	trainMat = []
	trainClasses = []

	for docIndex in trainingSet:
		trainMat.append(bagOfWords2Vec(vocabList, docList[docInded]))
		trainClasses.append(classList[docIndex])

	p0v, p1V, pSpam = trainNBO(np.array(trainMat), np.array(trainClasses))

	errorCount = 0

	for docIndex in testSet:
		wordVector = bagOfWords2Vec(covabList, docList[docIndex])

		if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
			errorCount += 1

	print('the error rate is: ', float(errorCOunt/len(testSet)))