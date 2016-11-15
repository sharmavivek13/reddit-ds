'''------------------------------------------------------------------------------
Basic Reddit Score Predictor by: Vivek Sharma, (sharma_vivek@icloud.com)

DataSet Used : Reddit Top 2.5 Million(https://github.com/umbrae/reddit-top-2.5-million) 
(Reference from Kaggle word2vec)
For More Info Refer to Readme.md
------------------------------------------------------------------------------'''

#Importing Required Modules
import io
import re
import nltk
import numpy as np
import requests
import pandas as pd
from flask import Flask
from flask import request
from flask import render_template
from nltk.corpus import stopwords
from gensim.models import Word2Vec 
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.preprocessing import Imputer

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template("my-form.html")

@app.route('/', methods=['POST']) 
def my_form_post():

	'''Reading input from Form and Getting required data from Github'''

	title=request.form['title']
	subreddit=request.form['subreddit']
	post=request.form['post']
	name=subreddit+".csv"
	url="https://raw.githubusercontent.com/umbrae/reddit-top-2.5-million/master/data/"+name
	s=requests.get(url).content	
	

###############################################################################

	'''Required Functions'''

	def makeFeatureVec(words, model, num_features):
            featureVec = np.zeros((num_features,),dtype="float32")
            nwords =0
            index2word_set = set(model.index2word)
            for word in words:
                if word in index2word_set:
                    nwords = nwords + 1.
                    featureVec = np.add(featureVec,model[word])
            featureVec = np.divide(featureVec,nwords)
            return featureVec

	def getAvgFeatureVecs(reviews, model, num_features):
            counter = 0
            reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
            for review in reviews:
                reviewFeatureVecs[counter] = makeFeatureVec(review, model,num_features)
                counter = counter + 1
            return reviewFeatureVecs

	def getCleanReviews(reviews):
            clean_reviews=[]
            for review in reviews:
                clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review))
            return clean_reviews

###############################################################################


	'''Training'''

	train = pd.read_csv(io.StringIO(s.decode('utf-8')),header=0)
	num_title = train["title"].size
	train_sentence=[None]*(num_title-1)
	for i in range(0,num_title-1):
            if(train["selftext"][i]=="nan"):
                train_sentence[i]=str(train["title"][i])+str(train["selftext"][i])
            else:
                train_sentence[i]=str(train["title"][i])
	tokenizer = nltk.data.load('tokenizers/english.pickle')
	sentences = []
	for review in train_sentence:
		sentences += KaggleWord2VecUtility.review_to_sentences(str(review),tokenizer)
	
###############################################################################

	model=Word2Vec(sentences,workers=4,size=30,min_count=8,seed=1,sample = 1e-3)
	model.init_sims(replace=True)

	trainDataVecs = getAvgFeatureVecs(getCleanReviews(str(train_sentence)), model, 30)
	trainDataVecs = Imputer().fit_transform(trainDataVecs)
        #Testing
	test_sentence=title+post
	testDataVecs = getAvgFeatureVecs(getCleanReviews(str(test_sentence)), model,30)
	testDataVecs = Imputer().fit_transform(testDataVecs)
	forest = RandomForestRegressor()
	forest = forest.fit(trainDataVecs, train["score"])
####
####
####
####	result = forest.predict(testDataVecs)
	output=str(-1)
	return output
	

if __name__ == '__main__':
    app.run(debug=True)
