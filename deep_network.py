import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time
import warnings
from scipy import sparse
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
from wordcloud import WordCloud ,STOPWORDS
from PIL import Image
import string
import re
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer   
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud,STOPWORDS
import warnings
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore")

from google.colab import files
uploaded = files.upload()
train_df =pd.read_csv('train.csv',error_bad_lines=False, engine="python")

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

#def cleanHtml(sentence):
    #cleanr = re.compile('<.*?>')
    #cleantext = re.sub(cleanr, ' ', str(sentence))
    #return cleantext

#def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
 #   cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
 #   cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
 #   cleaned = cleaned.strip()
 #   cleaned = cleaned.replace("\n"," ")
 #   return cleaned

def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

train_df['Message'] = train_df['Message'].str.lower()
#train_df['Message'] = train_df['Message'].apply(cleanHtml)
#train_df['Message'] = train_df['Message'].apply(cleanPunc)
train_df['Message'] = train_df['Message'].apply(keepAlpha)

#nltk.download('stopwords')
#stop_words = set(stopwords.words('english'))
stop_words = set(['between', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'such', 'into', 'of', 'most', 'itself', 'other', 'off',  's',  'or',  'as', 'from',  'each', 'the', 'until', 'below', 'these', 'through', 'don', 'nor',  'more', 'this', 'down', 'should',  'while', 'above', 'both', 'up', 'to',  'had', 'all', 'no', 'at',  'before',  'same', 'and', 'been', 'have', 'in', 'will', 'on',  'then', 'that', 'over',  'so', 'can', 'did', 'not', 'now', 'under', 'has', 'just', 'too', 'only', 'those',  'after', 'few',  't', 'being', 'if',  'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'])
#person: I/me or she/her
#interrogative sentence

stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

train_df['Message'] = train_df['Message'].apply(removeStopWords)

stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

train_df['Message'] = train_df['Message'].apply(stemming)

stopwords= set(['br', 'the',\
             'it', 'its', 'itself', \
             'this', 'that','these', 'those', \
            'be', 'been', 'being', 'had', 'having', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'all','both', 'each', 'few', 'more',\
            'most', 'other', 'such', 'only', 'own', 'same', 'so', 'than', 'very', \
            's', 't', 'will', 'just','now', 'd', 'o', \
            've', 'y'])
import re
def decontracted(phrase):
# specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

import itertools
from bs4 import BeautifulSoup
from tqdm import tqdm
preprocessed_comments = []
for sentence in tqdm(train_df['Message'].values):
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = BeautifulSoup(sentence, 'lxml').get_text()
    sentence = decontracted(sentence)
    sentence = re.sub("\S*\d\S*", "", sentence).strip()
    sentence = re.sub('[^A-Za-z]+', ' ', sentence)
    sentence = ''.join(''.join(s)[:2] for _, s in itertools.groupby(sentence))
    sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords)
    preprocessed_comments.append(sentence.strip())


count_vect = CountVectorizer() 
count_vect.fit(preprocessed_comments)
final_counts = count_vect.transform(preprocessed_comments)
count_vect = CountVectorizer(ngram_range=(1,2), min_df=10, max_features=5000)
final_bigram_counts = count_vect.fit_transform(preprocessed_comments)

tf_idf_vect = TfidfVectorizer(ngram_range=(1,2), min_df=10)
#tf_idf_vect = TfidfVectorizer()
tf_idf_vect.fit(preprocessed_comments)
final_tf_idf = tf_idf_vect.transform(preprocessed_comments)

text_col = ['Message']
label_col = [col for col in train_df.columns if col not in text_col]
#print(label_col)
final_tf_idf = tf_idf_vect.transform(preprocessed_comments)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import scipy
#split
features = text_col
X_test = test_df[features].copy()
X_train, X_val, y_train, y_val = train_test_split(train_df[features], train_df[label_col], test_size=0.2, random_state=2021)
X_train = tf_idf_vect.transform(X_train['Message'])
X_val = tf_idf_vect.transform(X_val['Message'])
X_test = tf_idf_vect.transform(X_test['Message'])
feature_names = tf_idf_vect.get_feature_names()

#print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

from sklearn.metrics import log_loss, roc_auc_score
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import layers

input_dim = X_train.shape[1] 
print(input_dim)

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
model.summary()
#X_train=np.asarray(X_train).astype(np.float32) 
#y_train=np.asarray(y_train).astype(np.float32)

X_train = np.array(X_train, dtype=object)
y_train = np.array(y_train, dtype=object)
history = model.fit(X_train, y_train,
                    epochs=100,
                    verbose=False,
                    validation_data=(X_val, y_val),
                    batch_size=10)
