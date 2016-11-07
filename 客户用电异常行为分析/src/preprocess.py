import xgboost as xgb
import numpy as np
import pandas as pd
import datetime
from sklearn.datasets import dump_svmlight_file
# read in data

def HandleDate(date):
	dateList = []
	for d in date:
		time = datetime.datetime.strptime(d, '%Y/%m/%d')
		baseDay = datetime.datetime.strptime("2015/01/01", '%Y/%m/%d')
		dateList.append((time - baseDay).days)
	return dateList



def translateData():

	dTrain = pd.read_csv("../data/train.csv", names=['CONS_NO', 'LABEL'])
	dTest = pd.read_csv('../data/test.csv', names=['CONS_NO'])

	dTrain.to_csv('../data/dTrain.csv', index = False)
	dTest.to_csv('../data/dTest.csv', index = False)


	dAllInformation = pd.read_csv("../data/all_user_yongdian_data_2015.csv", header = 0)
	date = HandleDate(dAllInformation['DATA_DATE'].values)
	dAllInformation['Time'] = date

	dAllInformation.to_csv('../data/userInfo.csv', index = False)


def getColName(day, stri):
	nameList = []
	for i in range(int(day)):
		print ('type i: ', type(i))
		nameList.append(stri + str(i))
	return nameList

def ConstructUserFeatureMatrix():

	UserInfo = pd.read_csv('../data/userInfo.csv')
	data = 	UserInfo[['CONS_NO', 'Time', 'KWH']].values
	userNum = len(np.unique(UserInfo['CONS_NO'].values))

	l = 1
	columns = 365/l

	matrix = np.zeros([userNum, columns + 1]) - 1
	userDict = {}

	UserNumInMatrix = 0

	for line in data:
		if line[0] not in userDict:
			userDict[line[0]] = UserNumInMatrix
			matrix[UserNumInMatrix, columns] = line[0]
			UserNumInMatrix += 1

		dateCol = line[1]

		if matrix[userDict[line[0]], dateCol] == -1:
			matrix[userDict[line[0]], dateCol] = line[2]
		else:
			matrix[userDict[line[0]], dateCol] += line[2]

	print ('type columns: ', type(columns))
	matrixColName = getColName(columns, 'Day ')
	matrixColName.append('CONS_NO')

	UserFeatureMatrix = pd.DataFrame(matrix, columns = matrixColName)
	name = '../data/UserFeatureMatrix.csv'
	UserFeatureMatrix.to_csv(name, index = False)


def getParameters():

	train = pd.read_csv('../data/dTrain.csv', header = 0)
	test = pd.read_csv('../data/dTest.csv', header = 0)
	UserFeatureMatrix = pd.read_csv('../data/UserFeatureMatrix.csv', header = 0)
	
	train = pd.merge(train, UserFeatureMatrix, on = 'CONS_NO', how = 'left').fillna(-1)
	test = pd.merge(test, UserFeatureMatrix, on = 'CONS_NO', how = 'left').fillna(-1)
	print ('train and test\'s shape:', train.shape, test.shape)
	
	"""
	train = pd.merge(train, UserFeatureMatrix, on = 'CONS_NO', how = 'inner').fillna(-1)
	test = pd.merge(test, UserFeatureMatrix, on = 'CONS_NO', how = 'inner').fillna(-1)
	print ('train1 and test1\'s shape:', train.shape, test.shape)
	"""

	trainFeature = train.drop(['CONS_NO', 'LABEL'], axis = 1)
	testFeature = test.drop(['CONS_NO'], axis = 1)

	trainLabel = train['LABEL'].values


	#trainFeature_opt = trainFeature.drop(trainFeature.columns[range(0,100)], axis = 1)
	#testFeature_opt = testFeature.drop(testFeature.columns[range(0,100)], axis = 1)
	#return trainFeature_opt, testFeature_opt, trainLabel, test['CONS_NO'].values

	return trainFeature, testFeature, trainLabel, test['CONS_NO'].values




