import pandas as pd
import numpy as np
import time 

global_start = time.time()

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
valid = pd.read_csv("Valid_rev1.csv", converters = converters)
print "Done!"


# Incomplete vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

def remove_bad_predictor(Dataframe, predictors):
	for i in len(predictors):
		Dataframe.drop(predictors[i], axis = 1, inplace=True)
	return Dataframe

predictors = ["SalaryRaw", "LocationRaw"]

# Incomplete ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

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

import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences(review, tokenizer, remove_stopwords=False):
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    #Returns a list of sentences, where each sentence is a list of words
    return sentences
   
sentences = []  # Initialize an empty list of sentences

print "Parsing sentences from training set"
for review in train["FullDescription"]:
    sentences += review_to_sentences(review.decode("utf8"), tokenizer)

print "Parsing sentences from test set"
for review in test["FullDescription"]:
    sentences += review_to_sentences(review.decode("utf8"), tokenizer)

print "Parsing sentences from validation set"
for review in valid["FullDescription"]:
    sentences += review_to_sentences(review.decode("utf8"), tokenizer)
# Should probably make script to remove URLs from reviews. 
print "..."
print "Done Parsing Sentences!"

# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 30   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words


print "Train Montage of word2vec..."
print "Queue Eye of the tiger"
# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print "Training model..."

model = word2vec.Word2Vec(sentences, workers=num_workers, 
                          size=num_features, min_count = min_word_count,
                          window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_30minwords_10context"
model.save(model_name)    
