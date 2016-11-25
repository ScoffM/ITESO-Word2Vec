import re
import nltk.data
import numpy as np
import random as ra
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#Loading the differents sets of data.
train = pd.read_csv( "labeledTrainData.tsv", header=0,
 delimiter="\t", quoting=3, encoding="utf-8" )
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3, encoding="utf-8" )
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0,
 delimiter="\t", quoting=3, encoding="utf-8" )

#Getting the reviews for each set of data.
x_train = train['review']
x_test = test['review']
x_unlabeled = unlabeled_train['review']

#Getting the sentiment just for train. In unlabeled_train we don't
#have sentiment column as same in test.
y_train = train['sentiment']

#Function to clean the data.
#Removes html, special characters like punctuation and put all the words in lower case
def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    clean_reviews = []
    for i in range(len(review)):
    # 1. Remove HTML
        review_text = BeautifulSoup(review[i]).get_text()
    #
    # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
        words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
    #
        clean_reviews.append(words)
    # 5. Return a list of reviews
    return(clean_reviews)

#Cleaning the data with the reviews in each set of data
x_train = review_to_wordlist(x_train)
x_test = review_to_wordlist(x_test)
x_unlabeled = review_to_wordlist(x_unlabeled)

#Function that labelize each review
def labelizeReviews(reviews, label_type):
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

x_concatenate = np.concatenate((x_train, x_unlabeled))

#Doing labilize procces to each data set, specifying in the parameter "label_type"
#what kind of data is the given data.
x_train2 = labelizeReviews(x_concatenate, 'TRAIN')
x_test2 = labelizeReviews(x_test, 'TEST')

#Doc2Vec
size = 50
model = Doc2Vec(min_count=1, window=5, size=size, sample=1e-4, negative=5, workers=8)

#build vocab over all reviews
model.build_vocab(x_train2)

#Training model
ra.seed(12345)
alpha = 0.025
min_alpha = 0.001
num_epochs = 5
alpha_delta = (alpha - min_alpha) / num_epochs

for epoch in range(num_epochs):
    print(epoch)
    ra.shuffle(x_train2)
    model.alpha = alpha
    model.min_alpha = alpha
    model.train(x_train2)
    alpha -= alpha_delta

#Infering over the test reviews in base the previous model built
test_vectors = np.zeros((len(x_test),size))
for i in range(len(x_test)):
    test_vectors[i] = model.infer_vector(x_test[i])

#Getting the train vectors from the model with the function docvecs.
#Remember that when we concatenate train and unlabeled sets, the first
#25,000 rows are from train set. The rest 50,000 rows are from the unlabeled set.
train_vectors = np.zeros((len(x_train), size))
for i in range(len(x_train)):
    train_vectors[i] = model.docvecs[i]

#model.save('my_model.doc2vec')

lgr = LogisticRegression()
lgr = lgr.fit(train_vectors, train["sentiment"])
results = lgr.predict(test_vectors)
output = pd.DataFrame(data={"id":test["id"], "sentiment":results})
output.to_csv( "Doc2Vec_lgr.csv", index=False, quoting=3 )