import xgboost as xgb
import numpy as np
import pandas as pd
import datetime
from sklearn.datasets import dump_svmlight_file
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

def TrainAndTest(trainFeature, testFeature, trainLabel, testLabel, params, rounds):

	dtrain = xgb.DMatrix(trainFeature, label = trainLabel)
	dtest = xgb.DMatrix(testFeature, label = testLabel)

	# specify parameters via map
	bst = xgb.train(params, dtrain, rounds, verbose_eval = 200)
	# make prediction
	preds = bst.predict(dtest)
	return preds

def modelfit(alg, params, trainFeature, trainLabel,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        """
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(trainFeature, label=trainLabel)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval = 10)
        alg.set_params(n_estimators=cvresult.shape[0])
        """
        xgtrain = xgb.DMatrix(trainFeature, label=trainLabel)
        cvresult = xgb.cv(alg.get_xgb_params(), xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            early_stopping_rounds=early_stopping_rounds, verbose_eval = 10)


    """
    #Fit the algorithm on the data
    alg.fit(trainFeature,trainLabel,eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(trainFeature)
    dtrain_predprob = alg.predict_proba(trainFeature)[:,1]
        
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(trainLabel, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(trainLabel, dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()
    """

def fix_parameters(param_test, trainFeature, trainLabel):

	gsearch = GridSearchCV(
		estimator = XGBClassifier(
			learning_rate=0.1, n_estimators=160, max_depth=7,
			min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8,
			objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27
		), 
	 	param_grid = param_test,
	 	scoring='roc_auc',
	 	n_jobs=4,
	 	iid=False,
	 	cv=5
	)

	gsearch.fit(trainFeature,trainLabel)

	print (gsearch.grid_scores_)
	print (gsearch.best_params_)
	print (gsearch.best_score_)

def xgbCVModel(trainFeature, trainLabel, testFeature, rounds, folds, params):
	
	#--Set parameter: scale_pos_weight-- 
	params['scale_pos_weight'] = (float)(len(trainLabel[trainLabel == 0]))/len(trainLabel[trainLabel == 1])
	print (params['scale_pos_weight'])


	#--Get User-define DMatrix: dtrain--
	#print trainQid[0]
	dtrain = xgb.DMatrix(trainFeature, label = trainLabel)
	num_round = rounds

	#--Run CrossValidation--
	print ('run cv: ' + 'round: ' + str(rounds) + ' folds: ' + str(folds))
	res = xgb.cv(params, dtrain, num_round, nfold = folds, early_stopping_rounds=50 , verbose_eval = 20)
	return res

