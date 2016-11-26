from gensim.models import Word2Vec
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import numpy as np

import time

master_start = time.time()

def identity(x):
    return x

# This allegedly increases speed in loading as it tells pandas to load thos oclumns as strings
converters = { "FullDescription" : identity
             , "Title": identity
             , "LocationRaw": identity
             , "LocationNormalized": identity
             }

print "Loading Data..."
train = pd.read_csv("Train_rev1.csv", converters = converters)
test =  pd.read_csv("Test_rev1.csv",  converters = converters)

model = Word2Vec.load("300features_30minwords_10context")

def description_to_wordlist(full_Description, remove_stopwords=False ):
    # 1. Remove HTML
    desc_text = BeautifulSoup(full_Description,"html.parser").get_text()
    # 2. Remove non-letters
    desc_text = re.sub("[^a-zA-Z]"," ", desc_text)
    # 3. Convert words to lower case and split them
    words = desc_text.lower().split()
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    # 5. Return a list of words
    return(words)

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")  # Pre-initialize an empty numpy array
    nwords = 0.
    index2word_set = set(model.index2word)  # Set of index2word, for speed
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(postings, model, num_features):
    # Take list of postings, get the feature vector and make it a 2D array
    counter = 0.
    postingFeatureVecs = np.zeros((len(postings),num_features),dtype="float32") 
    for posting in postings:
        # Print a status message every 1000th posting
        if counter%1000. == 0.:
            print "posting %d of %d" % (counter, len(postings))
        postingFeatureVecs[counter] = makeFeatureVec(posting, model, num_features)
        counter = counter + 1.
    return postingFeatureVecs

# Hopefully this works...
num_features = model.syn0.shape[1]

print "Creating average feature vecs for train postings"

clean_train_postings = []
for posting in train["FullDescription"]:
    clean_train_postings.append(description_to_wordlist(posting, remove_stopwords=True))

trainDataVecs = getAvgFeatureVecs(clean_train_postings, model, num_features)
train_new = pd.concat([pd.DataFrame(data=trainDataVecs), pd.DataFrame(train["SalaryNormalized"])],axis=1, join='inner')
train_new.to_csv("Train_Average.csv", index=False, quoting=3)

# Delete Variables to Save Memory? 
del train 
del trainDataVecs

print "Creating average feature vecs for test postings"



clean_test_postings = []
for posting in test["FullDescription"]:
    clean_test_postings.append(description_to_wordlist(posting, remove_stopwords=True))

testDataVecs = getAvgFeatureVecs(clean_test_postings, model, num_features)
test_new = pd.DataFrame(data= testDataVecs)
test_new.to_csv( "Test_Average.csv", index=False, quoting=3 )

print "Done!"
master_end = time.time()

master_elapsed = (master_end - master_start)/60

print "Time to build ", master_elapsed, "minutes"

