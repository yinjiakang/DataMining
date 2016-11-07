#-- encoding:UTF-8 --
import pandas as pd 
import numpy as np 
import datetime

def getDate(date):
	listTime = []
	for d in date:
		time = datetime.datetime.strptime(d,"%Y/%m/%d")
		time1 = datetime.datetime.strptime('2015/01/01',"%Y/%m/%d")
		listTime.append((time-time1).days)
	return listTime

def translateData():

	train = pd.read_csv('../data/train.csv', header = None)
	train.columns = ['CONS_NO','label']

	train.to_csv('../data/trainInfo.csv', index = False)

	test = pd.read_csv('../data/test.csv', header = None)
	test.columns = ['CONS_NO']

	test.to_csv('../data/testInfo.csv', index = False)

	'''
	useData	= pd.read_csv('../data/all_user_yongdian_data_2015.csv', header = 0)
	time = getDate(useData['DATA_DATE'].values)
	useData['Time'] = time

	useData.to_csv('../data/useDataInfo.csv', index = False)

	'''

def getColName(colNum, stri):
	print colNum, stri
	colName = []
	for i in range(colNum):
		colName.append(stri + str(i))
	return colName


def getUseMatrix(config):
	useData = pd.read_csv('../data/useDataInfo.csv', header = 0)
	data = useData[['CONS_NO','Time','KWH']].values

	userNum = len(np.unique(useData['CONS_NO'].values))

	for l in config['listMatrix']:
		print l
		timeNum = 365/l 
		print timeNum
		matrix = np.zeros([userNum, timeNum + 1]) - 1

		userDict = {}
		num = 0
		for line in data:
			if userDict.has_key(line[0]) == False:
				userDict[line[0]] = num
				matrix[userDict[line[0]], timeNum] = line[0]
				num += 1
			col = line[1]/l
			if matrix[userDict[line[0]], col] == -1:
				matrix[userDict[line[0]], col] = line[2]
			else:
				matrix[userDict[line[0]], col] += line[2]
		

		matrixColName = getColName(timeNum, 'useDay'+str(l)+'-')
		matrixColName.append('CONS_NO')

		matrixFeature = pd.DataFrame(matrix, columns = matrixColName)
		name = '../data/matrixFeature'+str(l)+'.csv'
		matrixFeature.to_csv(name, index = False)

def getDescriptionFeature(config):
	return 0


def getCountFeature(listMatrix):

	useMatrix = pd.read_csv('../data/matrixFeature.csv', header = 0)
	


def getFeature(config):
	train = pd.read_csv('../data/trainInfo.csv', header = 0)
	test = pd.read_csv('../data/testInfo.csv', header = 0)
	print train.shape, test.shape

	if config['useMatrix'] == True:
		for l in config['listMatrix']:
			name = '../data/matrixFeature'+str(l)+'.csv'
			useMatrix = pd.read_csv(name, header = 0)
			train = pd.merge(train, useMatrix, on = 'CONS_NO', how = 'left').fillna(-1)
			test = pd.merge(test, useMatrix, on = 'CONS_NO', how = 'left').fillna(-1)
			print train.shape, test.shape

	trainFeature = train.drop(['CONS_NO','label'], axis = 1)
	testFeature = test.drop(['CONS_NO'], axis = 1)

	trainLabel = train['label'].values

	print trainFeature.shape, testFeature.shape, trainLabel.shape

	return trainFeature, testFeature, trainLabel, test['CONS_NO'].values



def storeResult(testIndex, predict, model, day):
	result = pd.DataFrame({'CONS_NO':testIndex.T, 'label':predict})
	#print result
	rpath = '../result/'+ day + '.csv'
	mpath = '../model/'+ day + '.m'
	result = result.sort('label', ascending = False)
	#print result
	result = result['CONS_NO']
	result.to_csv(rpath, index = False, header = False)
	if model != False:
		model.dump_model(mpath)	










