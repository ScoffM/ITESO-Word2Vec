# ITESO-Word2Vec
Team development of Word2Vec algorithm for Kaggle Competition

## Salaries Modules Order
1. Salaries_Word2Vec.py: 

  Loads Train, Test and Validation data from kaggle, and trains a Word2Vec model.
  
2. Salaries_Average_Feature_Vector.py:

  Takes the trained w2v model and does vector averages, creates a new_train.csv
  
3. Salary_XGB_CV.py:

  Loads new_train.csv, then proceeds to train and cross validate a boosted trees method through XGB library

## Current Results
	
1. Word2Vec: MAE 7173.8247 +/- 03.3957
2. Doc2Vec : MAE 8002.9410 +/- 129.9566
2. TFIDF_C : MAE xxxx.xxxx +/- xx.xxxx



  
