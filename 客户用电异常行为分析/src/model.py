# -- encoding:UTF-8 --

import numpy as np 
import pandas as pd 
import xgboost as xgb 
import pywFM
from sklearn.preprocessing import OneHotEncoder
from function import *
from scipy.sparse import hstack



def xgbLocalModel(trainFeature, testFeature, trainLabel, testLabel, params, rounds):
	params['scale_pos_weight'] = (float)(len(trainLabel[trainLabel == 0]))/len(trainLabel[trainLabel == 1])
	print params['scale_pos_weight']

	dtrain = xgb.DMatrix(trainFeature, label = trainLabel)
	dtest = xgb.DMatrix(testFeature, label = testLabel)

	watchlist  = [(dtest,'eval'), (dtrain,'train')]
	num_round = rounds
	print 'run local: ' + 'round: ' + str(rounds)
	model = xgb.train(params, dtrain, num_round, watchlist, verbose_eval = 20)#,feval = evalerror)

	predict = model.predict(dtest)

	return predict

def xgbCVModel(trainFeature, trainLabel, testFeature, rounds, folds, params):
	
	#--Set parameter: scale_pos_weight-- 
	params['scale_pos_weight'] = (float)(len(trainLabel[trainLabel == 0]))/len(trainLabel[trainLabel == 1])
	print params['scale_pos_weight']


	#--Get User-define DMatrix: dtrain--
	#print trainQid[0]
	dtrain = xgb.DMatrix(trainFeature, label = trainLabel)
	num_round = rounds

	#--Run CrossValidation--
	print 'run cv: ' + 'round: ' + str(rounds) + ' folds: ' + str(folds) 
	res = xgb.cv(params, dtrain, num_round, nfold = folds, verbose_eval = 20)
	return res


def xgbPredictModel(trainFeature, trainLabel, testFeature, params, rounds):

	dtrain = xgb.DMatrix(trainFeature, label = trainLabel)
	dtest = xgb.DMatrix(testFeature, label = np.zeros(testFeature.shape[0]))

	watchlist  = [(dtest,'eval'), (dtrain,'train')]
	
	params['scale_pos_weight'] = (float)(len(trainLabel[trainLabel == 0]))/len(trainLabel[trainLabel == 1])

	print params['scale_pos_weight']

	num_round = rounds
	
	model = xgb.train(params, dtrain, num_round, watchlist, verbose_eval = 100)


	predict = model.predict(dtest)

	return model, predict




def pywfmLocalModel(trainFeature, testFeature, trainLabel, testLabel, trainIndex, testIndex, fm, cvIndex):


	print 'run local: folds: ' + str(cvIndex) 

	trainIndex, testIndex, value1, value2 = getIntId(trainIndex, testIndex)
	encoder = OneHotEncoder(n_values=[value1, value2])
	trainIndex_encode = encoder.fit_transform(trainIndex)
	testIndex_encode = encoder.transform(testIndex)

	trainFeature = hstack((trainIndex_encode, trainFeature))
	testFeature = hstack((testIndex_encode, testFeature))

	'''
	for i in range(len(trainLabel)):
		if i == 0:
			trainLabel[i] = -1
	for i in range(len(testLabel)):
		if i == 0:
			testLabel[i] = -1
	'''
	model = fm.run(trainIndex_encode, trainLabel, testIndex_encode, testLabel)

	predict = model.predictions

	predict = np.array(predict, np.float)

	predict = (predict - np.min(predict))/(np.max(predict) - np.min(predict))


	return predict

def pywfmPredictModel(trainFeature, testFeature, trainLabel, trainIndex, testIndex, fm):


	print 'run online!'

	trainIndex, testIndex, value1, value2 = getIntId(trainIndex, testIndex)
	encoder = OneHotEncoder(n_values=[value1, value2])
	trainIndex_encode = encoder.fit_transform(trainIndex)
	testIndex_encode = encoder.transform(testIndex)

	trainFeature = hstack((trainIndex_encode, trainFeature))
	testFeature = hstack((testIndex_encode, testFeature))

	#print trainFeature

	'''
	for i in range(len(trainLabel)):
		if i == 0:
			trainLabel[i] = -1
	for i in range(len(testLabel)):
		if i == 0:
			testLabel[i] = -1
	'''
	testLabel = np.zeros((testFeature.shape[0]))
	model = fm.run(trainFeature, trainLabel, testFeature, testLabel)

	predict = model.predictions

	predict = np.array(predict, np.float)
	print np.max(predict), np.min(predict)

	#predict = (predict - np.min(predict))/(np.max(predict) - np.min(predict))


	return predict












