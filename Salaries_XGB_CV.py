import pandas as pd
import numpy as np
import time
import os 
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
test_y = test["SalaryNormalized"]
train_y = train["SalaryNormalized"]

del data # "Unloading" might help memory

print "Data Loaded. Processing..."


# Native interface implementation
import xgboost as xgb

xgtrain = xgb.DMatrix(train_x.values, train_y.values)
xgtest = xgb.DMatrix(test_x.values, test_y.values)

params = {
    'objective':'reg:linear',
    'max_depth':6,
    'silent':0,
    'eta':.3, 
    #'eval_metric': 'mae'
}

watchlist  = [(xgtest,'test'), (xgtrain,'train')]

num_rounds = 1000

#bst = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds  = 10)

bst = xgb.cv(params, xgtrain, num_rounds, nfold=3, metrics = {'mae'}, early_stopping_rounds  = 5, seed = 1)

print bst

print "Done!"
master_end = time.time()
master_elapsed = (master_end - master_start)/60
print "Time to build ", master_elapsed, "minutes"


