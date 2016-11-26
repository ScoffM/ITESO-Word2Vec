import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import time

# Read Data and Fix it
data = pd.read_csv("cs-training.csv").drop('Unnamed: 0', axis = 1)


# Adjust Debt Ratio cap to 100%
def NorDebt(Debt_Ratio):
    if Debt_Ratio > 1:
        Debt_Ratio = 1
    else:
        Debt_Ratio = Debt_Ratio
    return Debt_Ratio

Debt_Ratio = data.DebtRatio.apply(NorDebt)
data = data.assign(Debt_Ratio=Debt_Ratio.values)
data.drop('DebtRatio', axis=1, inplace=True)

#Get rid of NA and nan
data.dropna(inplace = 'TRUE')


#Fix variables
data.ix[data.SeriousDlqin2yrs == 0, "SeriousDlqin2yrs"] = 0
data.ix[data.SeriousDlqin2yrs == 1, "SeriousDlqin2yrs"] = 1


#Split test, train
train, test = train_test_split(data)
coso = data.columns.values.tolist()
coso.remove('SeriousDlqin2yrs')
#coso.remove('ClassifierWeight')
train_x = train[coso]
train_y = train['SeriousDlqin2yrs'].tolist()
test_x = test[coso]
test_y = test['SeriousDlqin2yrs'].tolist()

def WeightDef (Class):
    if Class == 0:
        Weight = 1
    else:
        Weight = .933/.067
    return Weight

sample_weight = map(WeightDef, train_y)
test_weight = map(WeightDef, test_y)

from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV


vector = SVC(kernel = 'poly', class_weight = 'balanced')

param_grid = { 
    'C': [1,10,30,50,80,100],
    'degree': [3,4,5,6]
}

gridvec = GridSearchCV(estimator = vector, param_grid = param_grid, cv=5)

gridvec.fit(train_x, train_y)

print	gridvec.best_params_