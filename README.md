# Reddit Score Predictor

Basic Reddit Score Predictor made using bag of words and word2vec in python3.

Random Forests and Xgboost are used for prediction.

Program takes post title and text(if any) as input for specific subreddit and returns the expected score on reddit
Predicted score is not very accurate since the data is trained on post title and additional texts.
The new model with more features will be updated soon.

Run *web.py* for using * Bag of Words * and *web2.py* for *Word2Vec* model

Dataset used :Reddit Top 2.5 Million(https://github.com/umbrae/reddit-top-2.5-million)

Install the required modules of python using 
> sudo pip install -r requirements.txt

For using

* Go to the directory.
* Open terminal.
* enter "python web.py" 
* go to http://127.0.0.1:5000
