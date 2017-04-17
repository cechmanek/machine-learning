# plot decision tree using text anotations to simulate nodes & branches

import matplotlib.pyplot as plot

decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrowArgs = dict(arrowstyle='<-')

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
							xytext=centerPt, textcoords='axes fraction',
							va='center', ha='center', bbox=nodeType, arrowprops=arrowArgs)

def createPlot():
	fig = plot.figure(1, facecolor='white')
	fig.clf()
	createPlot.ax1 = plot.subplot(111, frameon=False)
	plotNode('a decion node', (0.5, 0.1), (0.1, 0.5), decisionNode)
	plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
	plot.show() 

def getNumLeafs(myTree):
	numLeafs = 0
	#firstStr = myTree.keys()[0]
	firstStr = list(myTree.keys())[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		 # test if node is dict or leaf
		if type(secondDict[key]).__name__ == 'dict':
			numLeafs += getNumLeafs(secondDict[key])
		else:
			numLeafs += 1
	
	return numLeafs

def getTreeDepth(myTree):
	maxDepth = 0
	#firstStr = myTree.keys()[0]
	firstStr = list(myTree.keys())[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			thisDepth = 1 + getTreeDepth(secondDict[key])
		else:
			thisDepth = 1

		if thisDepth > maxDepth:
			maxDepth = thisDepth

	return maxDepth

def retrieveTree(i):
	listOfTrees = [{'no surfacing':{0:'no', 1:{'flippers':{0:'no',1:'yes'}}}},
					{'no surfacing': {0:'no', 1:{'flippers':{0:{'head':{0:'no', 1:'yes'}},1:'no'}}}}]
	return listOfTrees[i]

def plotMidText(centerPt, parentPt, txtString):
	# plots tet between child and parent node
	xMid = (parentPt[0]-centerPt[0])/2.0 + centerPt[0]
	yMid = (parentPt[1]-centerPt[1])/2.0 + centerPt[1]
	createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
	numLeafs = getNumLeafs(myTree)
	#getTreeDepth(myTree)

	#firstStr = myTree.keys()[0]
	firstStr = list(myTree.keys())[0]
	secondDict = myTree[firstStr]

	centerPt = (plotTree.xOff + (1.0+float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
	plotMidText(centerPt, parentPt, nodeTxt) # plot child node
	plotNode(firstStr, centerPt, parentPt, decisionNode)
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD # decrement y offset

	for key in secondDict.keys():	
		if type(secondDict[key]).__name__ == 'dict':
			plotTree(secondDict[key], centerPt, str(key))
		else:
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
			plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), centerPt, leafNode)
			plotMidText((plotTree.xOff, plotTree.yOff), centerPt, str(key))
			
	plotTree.yOff = plotTree.yOff + 1.0/ plotTree.totalD

def createPlot(inTree):
	fig = plot.figure(1, facecolor='white')
	fig.clf()
	axProps = dict(xticks=[0.25,0.5,0.75], yticks=[0.25,0.5,0.75])

	createPlot.ax1 = plot.subplot(111, frameon=False, **axProps)
	plotTree.totalW = float(getNumLeafs(inTree))
	plotTree.totalD = float(getTreeDepth(inTree))

	plotTree.xOff = -0.5/plotTree.totalW
	plotTree.yOff = 1
	
	plotTree(inTree, (0.5,1.0),'')
	plot.show()


myTree = retrieveTree(0)
createPlot(myTree)

