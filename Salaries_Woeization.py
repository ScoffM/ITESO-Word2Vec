import pandas as pd
import numpy as np
import math
import time 

#colnames is a list of names of the columns to be transformed
#Should either:
#   a) Be ["LocationNormalized", "ContractType", "ContractTime", "Company", "Category", "SourceName"]
#   b) Pass data with only the above columns and use colnames.values
# The NaN's might have to be transformed before woeization can be completed. 
def woeization(data, target_variable, colnames):
    import numpy as np
    import math
    my_median = math.floor(data[target_variable].median())
    true_all = sum(data[target_variable] >= my_median)
    false_all = sum(data[target_variable] < my_median)

    for x in range(len(colnames)):
        xx = data[colnames[x]] # In each loop, set xx for an entire column 
        my_cat = np.unique(xx).tolist() # List of unique categories on my column xx
        for y in range(len(my_cat)):
            true = sum((xx == my_cat[y]) & (data[target_variable] >= my_median))
            false = sum((xx == my_cat[y]) & (data[target_variable] < my_median))
            # If the data is completely skewed towards a "side" 
            # Make it slightly larger than 0 to get out of the undefined zones of log(x) and 1/x
            if true == 0:  
                true = 0.001
            if false == 0:
                false = 0.001
            #Calcular WoE
            true_per = float(true) / true_all
            false_per = float(false) / false_all
            div = float(true_per) / false_per
            woe = math.log(div) 
            data.loc[data[colnames[x]] == my_cat[y], colnames[x]] = woe
    return data
        
# Run as standalone to get a modified dataframe, else import to get the modified features
def main():
    global_start = time.time()
    path = "Train_Rev1.csv"
    target_variable = "SalaryNormalized"
    colnames = list(data.columns.values)
    colnames = colnames[0:116]
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
    data_woe.to_csv('WoE_Features.csv')

if __name__=="__main__":
    main()

