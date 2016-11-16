'''------------------------------------------------------------------------------
Reddit Post Search by: Vivek Sharma, (sharma_vivek@icloud.com)

DataSet Used : Reddit Top 2.5 Million(https://github.com/umbrae/reddit-top-2.5-million) 
Program takes a word from user to be searched in a particular subreddit and returns 
all the posts in that subreddit containing the entered word.
Currently the results are ranked as per their score.
For More Info Refer to Readme.md
------------------------------------------------------------------------------'''

#Importing Required Modules

import io
import re
import itertools
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
	score=[]
	name=subreddit+".csv"
	url="https://raw.githubusercontent.com/umbrae/reddit-top-2.5-million/master/data/"+name
	s=requests.get(url).content
	train = pd.read_csv(io.StringIO(s.decode('utf-8')),header=0)
	num=train["title"].size
	for j in range(0,num):
		if word in train["title"][j]:
			main.append(train["title"][j])
			score.append(train["score"][j])
	list_enumerate = itertools.count()
	main.sort(reverse=True, key=lambda k: score[next(list_enumerate)])
	if not main:
		output="No Match Found"
		return output
	else:
		return render_template("temp.html",entries=main)

if __name__ == '__main__':
    app.run(debug=True)
