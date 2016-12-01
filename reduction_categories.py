import pandas as pd
import numpy as np
import math
from Salaries_Woeization import woeization

def eliminate_nan(col):
    trueNan = pd.isnull(col)
    indexs = trueNan[ trueNan == True].index.tolist()
    col[indexs] = 'True_nan'
    return col

def get_numberCategories(data):
    column_names = data.columns.values
    lenCategories_list = [0] * len(column_names)
    print "Number of categories in each feature"
    print " "
    for i in range(len(column_names)):
        print column_names[i]
        lenCategories_list[i] = len(np.unique(data[column_names[i]]))
        print lenCategories_list[i]
        print "..."
    return lenCategories_list

def Categories_reducction(data, Cat_names, target_var, reduction_percent):
    for i in range(len(Cat_names)):
        if len(np.unique(data[Cat_names[i]])) > 10:
            CatLen = len(np.unique(data[Cat_names]))
            n_newCats = math.ceil(CatLen * reduction_percent)
            binCats = 1 / n_newCats
            tableCats = data[target_var].groupby(data[Cat_names[i]]).median().sort_values()
            for j in range(int(n_newCats)):
                mini = j + 1
                q = tableCats.quantile(q=binCats * mini)
                true_Cats = tableCats[tableCats <= q].index.tolist()
                data[Cat_names[i]] = data[Cat_names[i]].replace(true_Cats, mini)
    #This function returns the same Data Frame but with the reduction in the number of categories
    return data

def main():
    #Loading data
    data = pd.read_csv("data/train_rev1.csv")
    target_cols = ["LocationNormalized", "ContractType", "ContractTime", "Category", "SourceName"]

    # Eliminating the nan values in each column from the df data
    for i in range(len(target_cols)):
        if data[target_cols[i]].isnull().values.any() == True:
            print target_cols[i]
            data[target_cols[i]] = eliminate_nan(data[target_cols[i]])

    CatsLen = get_numberCategories(data[target_cols])
    print "Starting with Categories reducction"
    #In this function we give just the feature "LocationNormalized" because it has more than 2,000 categories.
    #If you would like to include another future just add like in this example: [FeatureA, FeatureB, FeatureC]
    data = Categories_reducction(data, ['LocationNormalized'], 'SalaryNormalized', 0.1)

    print "Starting with features transformation from Categories into WoE"

    data_woe = woeization(data=data, target_variable='SalaryNormalized', colnames=target_cols)

    print "Writing csv with woe values "
    data_woe.to_csv('data/WoE_Features.csv')

if __name__=="__main__":
    main()