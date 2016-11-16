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
import requests
import pandas as pd
from flask import Flask
from flask import request
from flask import render_template


app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template("search.html")

@app.route('/', methods=['POST']) 
def my_form_post():


	word=request.form['word']
	subreddit=request.form['subreddit']
	main=[]
	name=subreddit+".csv"
	url="https://raw.githubusercontent.com/umbrae/reddit-top-2.5-million/master/data/"+name
	s=requests.get(url).content
	train = pd.read_csv(io.StringIO(s.decode('utf-8')),header=0)
	num=train["title"].size
	for j in range(0,num):
		if word in train["title"][j]:
			main.append(train["title"][j])
	output='\n'.join(map(str, main))
	return output

###############################################################################

if __name__ == '__main__':
    app.run(debug=True)
