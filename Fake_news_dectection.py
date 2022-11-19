#work flow 
#news data
#data pre-processing
#Train test split
#logiest regression model
#Build a system to identify unreliable news articles
# A full training dataset with the following attributes:

#id: unique id for a news article
#title: the title of a news article
#author: author of the news article
#text: the text of the article; could be incomplete
#label: a label that marks the article as potentially unreliable
#1: unreliable(fake)
#0: reliable(real)
#test.csv: A testing training dataset with all the same attributes at train.csv without the label.

#submit.csv: A sample submission that you can
#----------------------------------------install important labrary

import numpy as np#this also you know well
import pandas as pd#this you know 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score #this is used to find accurancy of our model 
import re #it is used to search word on paragraph
from nltk.stem.porter import PorterStemmer#it will gives root word in all words like loving into love
from nltk.corpus import stopwords #this is used for remove such word which not give importance to the paragrap
from sklearn.feature_extraction.text import TfidfVectorizer #this is used to convert data into vector form 

#-------------------------------------------------data pre-processing
import ast
import nltk
nltk.download('stopwords')
#print(stopwords.words("english"))#this is word which not important much
data = pd.read_csv("C:/Users\kunde/all vs code/ml prject/t.csv")
print(data.shape)
print(data.columns)
print(data.loc[0])
print(data.isnull().sum())
print(data.fillna(" "))
data["content"] = data['title']+data['author']#+data['text']
y = data["label"]

x_1 = data.drop(columns="label", axis=1)
x = x_1.drop(columns=['title', 'author'], axis=1)
print(x.columns)

#steming is process of reducing a word to its root word
port_stemer = PorterStemmer()
def steming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content= stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[port_stemer.stem(word) for word in stemmed_content if not word in stopwords.words("english")] 
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

x_1['content']=x_1['content'].apply(steming)
print(x_1.columns)


