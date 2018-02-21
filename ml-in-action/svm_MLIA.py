# support vector machine using sequential minimal optimization (smo)

import numpy as np

# defining simple helper functions

def loadDataSet(fileName):
	dataMat = []
	labelMat = []

	fr = open(fileName)

	for line in fr.readlines():
		lineArray = line.strip().split('\t')
		dataMat.append([float(lineArray[0]), float(lineArray[1])])
		labelMat.append(float(lineArray[2]))
		# data is 3 columns, tab separated, last column is label

	return dataMat, labelMat

def selectJrand(i,m):
	j = i
	while (j==i):
		j = int(np.random.uniform(0,m))

	return j

def clipAlpha(aj, H, L):

	if aj > H:
		aj = H

	if L > aj:
		aj = L

	return aj


# defining simplified SMO algorithm

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
	dataMatrix = np.mat(dataMatIn)
	labelMat = np.mat(classLabels).transpose()
	b = 0
	m,n = np.shape(dataMatrix)

	alphas = np.mat(np.zeros((m,1)))
	iteration = 0 
	while (iteration < maxIter):
		alphaPairsChanged = 0

		for i in range(m):
			fXi = float(np.multiply(alphas,labelMat).T * (dataMatrix*dataMatrix[i,:].T)) + b # our prediction of the class label
			Ei = fXi - float(labelMat[i]) # error of our prediction vs actual label

			if ((labelMat[i]*Ei< -toler) and (alphas[i]<C)) or ((labelMat[i]*Ei>toler) and (alphas[i]>0)):
				j = selectJrand(i,m) # randomly select second alpha to optimize along with first one
				fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[j,:].T)) + b # our prediction of the class label
				Ej = fXj - float(labelMat[j]) # error of prediction vs actual label (fxj vs labelMat[j])

				alphaIold = alphas[i].copy()
				alphaJold = alphas[j].copy()

				# make sure alphas stay between 0 and C, hence bounded optimization
				if (labelMat[i] != labelMat[j]):
					L = max(0, alphas[j] - alphas[i])
					H = min(C, C + alphas[j] - alphas[i])
				else:
					L = max(0, alphas[j] + alphas[i])					
					H = min(C, alphas[j] + alphas[i])
					
				if L==H:
					print("L==H")
					continue

				# calculate optimal amount to change alpha, which is eta
				eta = 2.0* dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
				
				if eta >= 0:
					print("eta>=0")
					continue				

				# update i by the same amound as j in the opposite direction
				alphas[j] -= labelMat[j]*(Ei - Ej)/eta
				alphas[j] = clipAlpha(alphas[j],H,L)

				if (abs(alphas[j] - alphaJold) < 0.00001):
					print("j not moving enough")
					continue

				alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
				
				# set constant term in hyperplane equation w.T*X + b
				b1 = b - Ei - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j] - alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
				b2 = b - Ej - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[j,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j] - alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T

				if (0 < alphas[i]) and (C > alphas[i]):
					b = b1
				elif (0 < alphas[j]) and (C > alphas[j]):
					b = b2
				else:
					b = (b1 + b2)/2.0

				alphaPairsChanged += 1
				print("iteration: %d: %d, pairs changed %d" %(iteration, i, alphaPairsChanged))

		if (alphaPairsChanged == 0):
			iteration += 1
		else:
			iteration = 0

		print("iteration number %d" % iteration)

	return b, alphas


# optimization methods to improve the smoSimple function abive

class optStruct: # defining a class to hold our values. could've been a dict, but this is a bit less typing
	def __init__(self, dataMatIn, classLabels, C, toler, kTup):
		self.X = dataMatIn
		self.labelMat = classLabels
		self.C = C
		self.tol = toler
		self.m = np.shape(dataMatIn)[0]
		self.alphas = np.mat(np.zeros((self.m, 1)))
		self.b = 0
		self.eCache = np.mat(np.zeros((self.m, 2))) # error cache
		self.K = np.mat(np.zeros((self.m, self.m)))
		for i in range(self.m):
			self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def calcEk(oS, k): # calculate error
	# fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X*oS.X[k,:].T)) + oS.b # predicted class
	# eK = fXk - float(oS.labelMat[k]) # predicted class - actual = error
	#return eK

	fXk = float(np.multiply(oS.alphas, oS.labelMat).T*oS.K[:,k] + oS.b)
	Ek = fXk - float(oS.labelMat[k])
	
	return Ek

def selectJ(i, oS, Ei): # heuristic to select next j
	maxK = -1
	maxDeltaE = 0
	Ej = 0

	oS.eCache[i] = [1, Ei]
	validEcacheList = np.nonzero(oS.eCache[:,0].A)[0] # returns list of indices of input that are nonzero

	if (len(validEcacheList)) > 1:
		for k in validEcacheList:
			if k == 1:
				continue

			Ek = calcEk(oS, k)
			deltaE = abs(Ei - Ek)

			if (deltaE > maxDeltaE):
				maxK = k
				maxDeltaE = deltaE
				Ej = Ek

		return maxK, Ej

	else:
		j = selectJrand(i, oS.m)
		Ej = calcEk(oS, j)
		
	return j, Ej

def updateEk(oS, k):
	Ek = calcEk(oS, k)
	oS.eCache[k] = [1, Ek]


# full SMO optimization routine from Platt paper

def innerL(i, oS):
	Ei = calcEk(oS, i)

	# second choice heuristic
	if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
		j, Ej = selectJ(i, oS, Ei)
		alphaIold = oS.alphas[i].copy()
		alphaJold = oS.alphas[j].copy()

		if (oS.labelMat[i] != oS.labelMat[j]):
			L = max(0, oS.alphas[j] + oS.alphas[i]) # lower bound
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i]) # higher bound
		else:
			L = max(0, oS.alphas[j] + oS.alphas[i] -oS.C)
			H = min(oS.C, oS.alphas[j] + oS.alphas[i])

		if L == H:
			print("L==H")
			return 0

		# calculate optimal amount to update alpha
		# eta = 2.0*oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
		eta = 2.0*oS.K[i,j] - oS.K[i,i] - oS.K[j,j]

		if eta >= 0:
			print("eta>=0")
			return 0

		oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
		oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
		updateEk(oS, j)

		if (abs(oS.alphas[j] - alphaJold) < 0.00001):
			print("j not moving enough")
			return 0

		oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])

		updateEk(oS, i)

		# b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j] - alphaJold)*oS.X[i,:]*oS.X[j,:].T
		# b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j] - alphaJold)*oS.X[j,:]*oS.X[j,:].T
		b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j] - alphaJold)*oS.K[i,j]
		b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.K[i,j] - oS.labelMat[j]*(oS.alphas[j] - alphaJold)*oS.K[j,j]

		if (0< oS.alphas[i]) and (oS.C > oS.alphas[i]):
			oS.b = b1
		elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
			oS.b = b2
		else:
			oS.b = (b1 + b2)/2.0

		return 1

	else:
		return 0

# full Platt SMO outer loop

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
	oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)

	iteration = 0

	entireSet = True
	alphaPairsChanged = 0
	while(iteration < maxIter) and ((alphaPairsChanged > 0)) or entireSet:
		alphaPairsChanged = 0

		if entireSet:
			for i in range(oS.m):
				alphaPairsChanged += innerL(i, oS)

			print("fullSet, iter: %d i: %d, pairs changed %d" % (iteration, i, alphaPairsChanged))
			iteration += 1
		else:
			nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]

			for i in nonBoundIs:
				alphaPairsChanged  += innerL(i, oS)
				print("non-bound, iter: %d i:%d, pairs changed %d" % (iteration, i, alphaPairsChanged))

			iteration += 1

		if entireSet:
			entireSet = False
		elif (alphaPairsChanged == 0):
			entireSet = True

		print( "iteration number %d" % iteration)

	return oS.b, oS.alphas


# now that the hyperplane alphas and b have been determined we can use them to classify

def calcWs(alphas, dataArray, classLabels):
	X = np.mat(dataArray)
	labelMat = np.mat(classLabels).transpose()
	m,n = np.shape(dataArray)
	w = np.zeros((n,1))

	for i in range(m):
		w += np.multiply(aplhas[i]*labelMat[i], X[i,:].T)

	return w

# add some kernel transformation function to deal with data that cannot be sperated via straight ling

def kernelTrans(X, A, kTup):
	m,n = np.shape(X)
	K = np.mat(np.zeros((m,1)))

	if kTup[0] == 'lin': # defualt kernel if not specified is linear
		K = K = X * A.T
	elif kTup[0] == 'rbf':
		for j in range(m):
			deltaRow = X[j,:] - A
			K[j] = deltaRow*deltaRow.T

		K = np.exp(K / (-1*kTup[1]**2)) # '/' is element-wise division in numpy
	else:
		raise NameError("unknown kernel type specified, use 'lin', or 'rbf'")
	
	return K

def testRBF(k1=1.3): # k1 is a parameter for the gaussian radial basis function
	dataArr, labelArr = loadDataSet('testSetRBF.txt')
	b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 1000, ('rbf', k1))

	dataMat = np.mat(dataArr)
	labelMat = np.mat(labelArr).transpose()

	svInd = np.nonzero(alphas.A > 0)[0] # support vector indices
	sVs = dataMat[svInd] # the actual support vectors
	labelSV = labelMat[svInd]

	print("there are %d support vectors" % np.shape(sVs)[0])

	m,n = np.shape(dataMat)
	errorCount = 0

	for i in range(m):
		kernelEval = kernelTrans(sVs, dataMat[i,:],('rbf', k1))
		predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b

		if np.sign(predict) != np.sign(labelArr[i]):
			errorCount += 1

	print("the training error rate is %f" % float(errorCount/m))

	# now test the trained model on a new sample of data
	dataArr, labelArr = loadDataSet('testSetRBF2.txt')
	errorCout = 0

	dataMat = np.mat(dataArr)
	labelMat = np.mat(labelArr).transpose()
	m,n = np.shape(dataMat)

	for i in range(m):
		kernelEval = kernelTrans(sVs, dataMat[i,:], ('rbf', k1))
		predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b

		if np.sign(predict) != np.sign(labelArr[i]):
			errorCount += 1

	print("the test error rate is %f" % float(errorCount/m))


# now use this svm for handwriting character recognition

def loadImages(dirName):
	from os import listdir
	hwLabels = []

	trainingFileList = listdir(dirName)
	m = len(trainingFileList)

	trainingMat = np.zeros((m,1024))

	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])

		if classNumStr == 9:
			hwLabels.append(-1)
		else:
			hwLabels.append(1)

		trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))

	return trainingMat, hwLabels

def img2vector(fileName):
	returnVect = np.zeros((1,1024))
	fr = open(fileName)

	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])

	return returnVect

def testDigits(kTup = ('rbf', 10)):
	dataArr, labelArr = loadImages('trainingDigits')
	b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 1000, kTup)
	dataMat = np.mat(dataArr)
	labelMat = np.mat(labelArr).transpose()

	svInd = np.nonzero((alphas.A > 0))[0] # support vector indices
	sVs = dataMat[svInd] # actual support vectors
	labelSV = labelMat[svInd]
	print(" there are %d support vectors" % np.shape(sVs)[0])

	m,n = np.shape(dataMat)
	errorCount = 0

	for i in range(m):
		kernelEval = kernelTrans(sVs, dataMat[i,:], kTup)
		predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
		
		if np.sign(predict) != np.sign(labelArr[i]):
			errorCount += 1

	print("the training error rate is %f " % float(errorCount/m))

	# now test the trained model on a second data set
	dataArr, labelArr = loadImages('testDigits')
	
	errorCount = 0
	dataMat = np.mat(dataArr)
	labelMat = np.mat(labelArr).transpose()
	m,n = np.shape(dataMat)

	for i in range(m):
		kernelEval = kernelTrans(sVs, dataMat[i,:], kTup)
		predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
		
		if np.sign(predict) != np.sign(labelArr[i]):
			errorCount += 1

	print("the test error rate is %f " % float(errorCount/m))

