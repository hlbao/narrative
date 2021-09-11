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
from google.colab import files
uploaded = files.upload()
test_df =pd.read_csv('test.csv',error_bad_lines=False, engine="python")

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext

def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[\'|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

train_df['Message'] = train_df['Message'].str.lower()
train_df['Message'] = train_df['Message'].apply(cleanHtml)
train_df['Message'] = train_df['Message'].apply(cleanPunc)
train_df['Message'] = train_df['Message'].apply(keepAlpha)

stop_words = {'ever', "don't", 'during', 'nor', 'where', "why's", 'yourself', 'not', 'very', 'against', 'between', 'up', 'over', "aren't", "shouldn't", 'could', 'the', "they'd", 'after', "haven't", "we're", 'these', 'them', 'cannot', 'once', 'shall', 'own', 'until', 'get', "couldn't", 'who', "that's", 'then', 'for', 'have', "you've", 'while', 'else', 'be', 'www', 'and', 'any', 'which', 'than', 'because', 'into', 'to', "when's", 'you', 'a', "can't", 'off', 'since', 'under', 'out', 'so', "there's", 'down', 'more', "we'd", "we've", 'again', 'only', "they're", 'your', 'yourselves', "you'll", 'like', "they'll", 'it', 'some', 'however', "mustn't", 'of', 'both', 'such', 'ought', 'can', 'how', 'should', 'just', 'having', 'itself', 'other', 'yours', "who's", 'r', 'do', 'whom', 'com', "it's", "how's", 'most', 'what', "what's", "wouldn't", 'each', 'been', 'but', 'our', "you're", 'has', 'k', 'those', "here's", 'also', 'there', 'themselves', 'is', 'if', 'in', "won't", 'too', 'theirs', 'from', "you'd", 'as', 'they', 'we', 'would', "doesn't", 'all', 'when', 'below', "where's", 'before', 'no', 'about', 'being', "we'll", 'at', "isn't", 'same', 'their', 'above', 'here', 'with', 'ourselves', "they've", 'are', "shan't", 'why', 'its', 'on', 'few', 'or', 'by', 'that', 'further', 'through', "hasn't", 'otherwise', 'does', 'this', 'ours', 'an', 'http'}
#persona
#interrogative sentence or imperative sentence

stop_words.update(['to', 'the', 'and', 'breast', 'cancer', 'of', 'a', 'in', 'for', 'your', 'you', 'is', 'with', 'we', 'this', 'our', 'on', 'that', 'are', 'by', 'women', 'be', 'about', 'from', 'can', 'or', 'us', 'will', 'have', 'as', 'help', 'who', 'cancer.', 'their', 'all', 'at', 'more', 'day', 'has', 'support', 'it', 'one', 'how', 'so', 'what', 'an', 'make', 'up', 'treatment', '–', 'out', 'people', 'like', 'now', 'been', 'every', 'but', 'some', 'diagnosed', 'cancer,', 'not', 'find', 'when', 'get', 'they', '-', 'would'])
stop_words.update(['and', 'to', 'the', 'a', 'of', 'breast', 'cancer', 'with', 'in', 'for', 'that', 'is', 'it', 'on', 'have', 'you', 'this', 'diagnosed', 'our', 'but', 'at', 'we', 'about', 'as', 'be', 'so', 'are', 'your', 'like', 'cancer.', 'has', 'when', 'from', 'not', 'all', 'by', 'who', 'been', 'an', 'more', 'will', 'how', 'day', 'treatment', 'what', 'would', 'can', 'us', 'one', 'support', 'cancer,', 'women', 'find', 'out', '-', 'help', 'their', 'they', '–', 'people', 'now', 'every', 'get', 'or', 'up', 'some', 'make'])
stop_words.update(['life', 'bit', 'people', 'women', 'https', 'org', 'treatment', 'help', 'support', 'breast', 'cancer'])

set1 = ['new', 'research', 'pink', 'share', 'those', 'run', 'could', 'if', 'want', 'awareness', 'tell', 'today', 'metastatic', 'know', 'early', 'over', 'canadian', 'it’s', 'live', 'nurse', 'affected', 'survivors', 'join', 'sign', 'register', 'which', 'may', 'chance', 'many', 'comments', 'why', 'learn', 'need']
for i in range(len(set1)):
  if((set1[i] in stop_words) == True):
    stop_words.remove(set1[i])

set2= ['i', 'was', 'my', 'her', 'she', 'had', 'after', 'me', 'through', 'years', 'just', 'am', 'life', 'no', 'were', 'family', 'time', 'during', 'going', 'felt', 'go', 'there', 'tips', 'being', 'hope', 'told', 'other', 'first', 'i’m', 'didn’t', 'only', 'found', 'chemo']
for i in range(len(set2)):
  if((set2[i] in stop_words) == True):
    stop_words.remove(set2[i])

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

stopwords= stop_words

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

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import log_loss, roc_auc_score
#model = MultinomialNB(alpha = 0.1)
model = LogisticRegression(C=12)
#from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier()

train_rocs = []
valid_rocs = []
preds_train = np.zeros(y_train.shape)
preds_valid = np.zeros(y_val.shape)
preds_test = np.zeros((len(test_df), len(label_col)))

for i, label_name in enumerate(label_col):
    print('\nClass:= '+label_name)
    # fit
    model.fit(X_train,y_train[label_name])
    # train
    preds_train[:,i] = model.predict_proba(X_train)[:,1]
    #y_predict = model.predict(X_train)
    train_roc_class = roc_auc_score(y_train[label_name],preds_train[:,i])
    print('Train ROC AUC:', train_roc_class)
    #print(classification_report(y_val,y_predict))
    train_rocs.append(train_roc_class)

    # valid
    preds_valid[:,i] = model.predict_proba(X_val)[:,1]
    #y_val_predict = model.predict(X_val)
    valid_roc_class = roc_auc_score(y_val[label_name],preds_valid[:,i])
    print('Valid ROC AUC:', valid_roc_class)
    valid_rocs.append(valid_roc_class)
    
    # test predictions
    preds_test[:,i] = model.predict_proba(X_test)[:,1]
    
print('\nmean column-wise ROC AUC on Train data: ', np.mean(train_rocs))
print('mean column-wise ROC AUC on Val data:', np.mean(valid_rocs))

#sub_df_mnb.iloc[:,1:] = preds_test
#np.savetxt("result_MultinomialNB.csv", preds_test, delimiter=",")
#files.download('result_MultinomialNB.csv')
#np.savetxt("result_Logistic_Regression.csv", preds_test, delimiter=",")
#files.download('result_Logistic_Regression.csv')
#np.savetxt("result_RandomForest.csv", preds_test, delimiter=",")
#files.download('result_RandomForest.csv')

