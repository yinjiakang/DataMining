import xgboost as xgb
import numpy as np
import pandas as pd
import datetime
from preprocess import *
from train import *
from sklearn.datasets import dump_svmlight_file
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

config = {
	'rounds': 5000
}

params={
	'booster' : 'gbtree',
	'objective' : 'binary:logistic',
	'eval_metric': 'map',

	'max_depth' : 7,
	'min_child_weight' : 5,
	'gamma':0.4,
	'colsample_bytree': 0.9,
	'subsample': 0.9,

	'eta': 0.02,


	'nthread':4,
	'silent' : 1
}

xgb_param = XGBClassifier(
	learning_rate =0.1,
	n_estimators=160,
	max_depth=7,
	min_child_weight=1,
	gamma=0.4,
	subsample=0.9,
	colsample_bytree=0.9,
	objective= 'binary:logistic',
	nthread=4,
	scale_pos_weight=5,
	seed=27
)

param_mdandmcw = {
	'max_depth': [3,5,7,9],
	'min_child_weight':[1,3,5]
	}

param_mdandmcw2 = {
	'max_depth':[6,7,8],
	'min_child_weight':[4,5,6]
}

param_gamma = {
	'gamma': [i / 10.0 for i in range(0,10)]
}

param_sample = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

if __name__ == '__main__':

	#translateData()
	#ConstructUserFeatureMatrix()

	trainFeature, testFeature, trainLabel, testIndex = getParameters()

	"""
	trainFeature.to_csv('../data/trainFeature.csv', index = False)
	testFeature.to_csv('../data/testFeature.csv', index = False)
	trainLabel.to_csv('../data/trainLabel.csv', index = False)
	testIndex.to_csv('../data/testIndex.csv', index = False)

	trainFeature = pd.read_csv('../data/trainFeature.csv', header = 0)
	testFeature = pd.read_csv('../data/testFeature.csv', header = 0)
	trainLabel = pd.read_csv('../data/trainLabel.csv', header = 0)
	testIndex = pd.read_csv('../data/testIndex.csv', header = 0)
 		"""

	#print (trainFeature)
	print ('\n\n')
	#print (testFeature)
	print ('\n\n')
	#print (trainLabel)

	print ('finish getparameters')


	#preds = TrainAndTest(trainFeature, testFeature, trainLabel, np.zeros(testFeature.shape[0]), params, config['rounds'])
	print ('trainandtest finish')




	#cross-validation
	modelfit(xgb_param, params, trainFeature, trainLabel)
	print ('modelfit finish')

	#cvmodel = xgbCVModel(trainFeature, trainLabel, testFeature, config['rounds'], 5, params)
	#print ('cvmodel finish')

	#parameter tuning
	#fix_parameters(param_sample, trainFeature, trainLabel)

	"""
	result = pd.DataFrame({'CONS_NO': testIndex.T, 'RESULT': preds})
	result = result.sort('RESULT', ascending = False)
	result = result['CONS_NO']

	result.to_csv('../data/result_v3_5000.csv', index = False, header = False)
	"""
