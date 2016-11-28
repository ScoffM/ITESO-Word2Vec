import pandas as pd
import numpy as np
import math
import time 

#Replace the nan values with the string True_nan in a dataframe's column
def eliminate_nan(col):
    trueNan = pd.isnull(col)
    indexs = trueNan[ trueNan == True].index.tolist()
    col[indexs] = 'True_nan'
    return col

#colnames is a list of names of the columns to be transformed
#Should either:
#   a) Be ["ContractType", "ContractTime", "Category", "SourceName"]
#   b) Pass data with only the above columns and use colnames.values
# The NaN's might have to be transformed before woeization can be completed.
#This function returns a dataframe with woe values just with the specified columns
def woeization(data, target_variable, colnames):
    import numpy as np
    import math
    my_median = math.floor(data[target_variable].median())
    true_all = sum(data[target_variable] >= my_median)
    false_all = sum(data[target_variable] < my_median)

    for x in range(len(colnames)):
        #If the column has any nan value, the nan function is applies
        if data[colnames[x]].isnull().values.any() == True:
            data[colnames[x]] = eliminate_nan(data[colnames[x]])

        xx = data[colnames[x]]  # In each loop, set xx for an entire column
        my_cat = np.unique(xx).tolist()  # List of unique categories on my column xx

        for y in range(len(my_cat)):
            true = sum((xx == my_cat[y]) & (data[target_variable] >= my_median))
            false = sum((xx == my_cat[y]) & (data[target_variable] < my_median))
            # If the data is completely skewed towards a "side"
            # Make it slightly larger than 0 to get out of the undefined zones of log(x) and 1/x
            if true == 0:
                true = 0.001
            if false == 0:
                false = 0.001
            # Calcular WoE
            true_per = float(true) / true_all
            false_per = float(false) / false_all
            div = float(true_per) / false_per
            woe = math.log(div)
            data.loc[data[colnames[x]] == my_cat[y], colnames[x]] = woe
    data = data[(colnames + [target_variable])]
    return data
        
# Run as standalone to get a modified dataframe, else import to get the modified features
def main():
    global_start = time.time()
    path = "data/Train_Rev1.csv"
    target_variable = "SalaryNormalized"
    colnames = ['ContractType', 'ContractTime', 'Category', 'SourceName']
    def identity(x):
        return x
    # This allegedly increases speed in loading as it tells pandas to load thos oclumns as strings
    converters = { "FullDescription" : identity
                 , "Title": identity
                 , "LocationRaw": identity
                 , "LocationNormalized": identity
                 }
    print "Loading Data..."
    data = pd.read_csv(path)
    print "Done!"
    print "Initializing Data Transformation"
    data_woe= woeization(data=data, target_variable=target_variable, colnames=colnames)
    data_woe.to_csv('data/WoE_Features.csv')

if __name__=="__main__":
    main()

