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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer


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
	tokenizer = nltk.data.load('tokenizers/english.pickle')

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

	def wordbank(raw_title):
		letters_only = re.sub("[^a-zA-Z]", " ", raw_title)
		words = letters_only.lower().split()
		stops = set(stopwords.words("english"))
		meaningful_words = [w for w in words if not w in stops]
		return( " ".join( meaningful_words ))

	def review_to_sentences(titlee,tokenizer,remove_stopwords=False):
		raw_sentences=tokenizer.tokenize(str(titlee).strip())
		sentences=[]
		for raw_sentence in raw_sentences:
			if len(raw_sentence) > 0:
				sentences.append(wordbank(raw_sentence))
		return sentences

###############################################################################


	'''Training'''

	train = pd.read_csv(io.StringIO(s.decode('utf-8')),header=0)
	num_title = train["title"].size
	train_sentence=[]
	train_sentence=train["title"]+train["selftext"]

	sentences = []
	for review in train_sentence:
		sentences += review_to_sentences(review,tokenizer)

###############################################################################

	model=Word2Vec(sentences,workers=4,size=300,min_count=8,window=10,sample=1e-3)
	model.init_sims(replace=True)


	trainDataVecs = getAvgFeatureVecs(wordbank(str(train_sentence)), model, 300)

	#Testing
	test_sentence=title+post
	testDataVecs = getAvgFeatureVecs(wordbank(str(test_sentence)), model, 300)

	forest = RandomForestClassifier(n_estimators = 100 )
	forest = forest.fit( trainDataVecs, train["score"] )



	result = forest.predict(testDataVecs)
	output=str(result)
	return output
	

if __name__ == '__main__':
    app.run(debug=True)
