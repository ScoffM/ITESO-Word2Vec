import pandas as pd
import numpy as np
import time
import os 
# from gensim.models import Word2Vec
# from bs4 import BeautifulSoup
# import re
# from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
master_start = time.time()

print "Loading Data..."
cwd = os.getcwd()
data = pd.read_csv(cwd +"\\Datos\\Train_Average.csv")
train, test = train_test_split(data, test_size=.2, random_state=0)

predictors = data.columns.values.tolist()
predictors.remove("SalaryNormalized")
train_x = train[predictors]
test_x = test[predictors]

# Swap test_y and train_y definitions to use sklearn interface
test_y = test["SalaryNormalized"]
# test_y = train["SalaryNormalized"].tolist()
train_y = train["SalaryNormalized"]
# train_y = train["SalaryNormalized"].tolist()

del data # "Unloading" might help memory

print "Data Loaded. Processing..."

# Train XGBoost with it

#import xgboost as xgb
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import ShuffleSplit
#from sklearn.metrics import mean_absolute_error, make_scorer

# Missing early stop booster
#booster = xgb.XGBRegressor()
#booster.fit(trainDataVecs, train["SalaryNormalized"])
#cv = ShuffleSplit(n_splits=5, test_size= 0.3, random_state=0)

#print "Starting CV xgboost"
#mae = make_scorer(mean_absolute_error, greater_is_better = True)

#fit_params = {
#    'eval_set' : [(test_x, test_y)],
#    'eval_metric': mae,
#    'early_stoping_rounds': 10
#}

#scores = cross_val_score(booster, train_x, train_y, cv=cv, scoring=mae, fit_params= fit_params)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#Test & extract results 
#result = booster.predict(testDataVecs)

#Write the test results 
#output = pd.DataFrame( data={"Id":test["Id"], "sentiment":result} )
#output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )


# Native interface implementation

import xgboost as xgb
xgtrain = xgb.DMatrix(train_x.values, train_y.values)
xgtest = xgb.DMatrix(test_x.values, test_y.values)

params = {
    'objective':'reg:linear',
    'max_depth':6,
    'silent':1,
    'eta':.3, 
    'eval_metric': 'mae'
}

watchlist  = [(xgtest,'test'), (xgtrain,'train')]

num_rounds = 1000

bst = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds  = 10)

print "Done!"
master_end = time.time()
master_elapsed = (master_end - master_start)/60
print "Time to build ", master_elapsed, "minutes"


