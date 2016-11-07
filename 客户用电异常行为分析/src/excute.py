#-- encoding:UTF-8 --
import numpy as np 
import pandas as pd 
import datetime
from function import *
from model import *


config = {
	'rounds':5000,
	'folds':3,
	'useMatrix':True,
	'listMatrix':[1],
	'description':True,
	'countFeature':False,
	}


params={
	'scale_pos_weight': 0,
	'booster':'gbtree',
	'objective': 'binary:logistic',
    'eval_metric': 'map',
	'stratified':True,

	'max_depth':3,
	'min_child_weight':1,
	'gamma':0.1,
	'subsample':0.7,
	'colsample_bytree':0.6,

	
	'lambda':1,   #550
	#'alpha':1,
	#'lambda_bias':0,
	
	'eta': 0.02,
	'seed':12,

	'nthread':4,
	
	'silent':1
}


if __name__ == '__main__':

	#translateData()
	#getUseMatrix(config)
	trainFeature, testFeature, trainLabel, testIndex = getFeature(config)

	res = xgbCVModel(trainFeature, trainLabel, testFeature, config['rounds'], config['folds'], params)

	#model, predict = xgbPredictModel(trainFeature, trainLabel, testFeature, params, config['rounds'])

	#storeResult(testIndex, predict, model, '2016-10-07-02')







	#print useData[useData['CONS_NO'] == 7958116557]






