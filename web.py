'''------------------------------------------------------------------------------
Basic Reddit Score Predictor by: Vivek Sharma, (sharma_vivek@icloud.com)

DataSet Used : Reddit Top 2.5 Million(https://github.com/umbrae/reddit-top-2.5-million) 
(Reference from Kaggle Bag of Words)
Two Models are used for prediction: Xgboost and Random Forest. By default random 
forest is selected.
To use xgboost uncomment lines no.76,77,78,92 and comment lines no. 81,82,91
For More Info Refer to Readme.md
------------------------------------------------------------------------------'''

#Importing Required Modules

import io
import re
import nltk
import numpy
import xgboost
import requests
import pandas as pd
from flask import Flask
from flask import request
from flask import render_template
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

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

	'''Training'''

	train = pd.read_csv(io.StringIO(s.decode('utf-8')),header=0)
	num_title = train["title"].size
	clean_sentences = []
	train_sentence=[None]*(num_title)
	for i in range(0,num_title):
		if(train["selftext"][i]=="nan"):
			train_sentence[i]=str(train["title"][i])+str(train["selftext"][i])
		else:
			train_sentence[i]=str(train["title"][i])
	def wordbank(raw_title):
		letters_only = re.sub("[^a-zA-Z]", " ", raw_title)
		words = letters_only.lower().split()
		stops = set(stopwords.words("english"))
		meaningful_words = [w for w in words if not w in stops]
		return( " ".join( meaningful_words ))
	for i in range(0, num_title ):
		clean_sentences.append(wordbank(str(train_sentence[i])))

	vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000) 
	train_data_feature=vectorizer.fit_transform(clean_sentences)
	train_data_feature=train_data_feature.toarray()

###############################################################################

	'''Model for Predicting Score'''

	# xgmodel=xgboost.XGBClassifier()
	# xgmodel=xgboost.XGBClassifier(n_estimators=130,max_depth=6,min_child_weight=1,gamma=0.1,learning_rate=0.01,nthread=4)
	# xgmodel=xgmodel.fit(train_data_feature, train["score"])

	# Random Forest Model
	forest = RandomForestClassifier(n_estimators = 120)
	forest = forest.fit(train_data_feature, train["score"])

###############################################################################

	'''Testing'''
	clean_test_sentences=[]
	clean_test_sentences=wordbank((title+post))
	test_data_feature = vectorizer.transform(clean_test_sentences)
	test_data_feature = test_data_feature.toarray()
	# result = xgmodel.predict(test_data_feature)
	result = forest.predict(test_data_feature)
	output="The predicted score is:"+str(result[0])
	return output

###############################################################################

if __name__ == '__main__':
    app.run(debug=True)
