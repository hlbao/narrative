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
np.random.seed(10000000)

from google.colab import files
uploaded = files.upload()
train_df =pd.read_csv('train.csv',error_bad_lines=False, engine="python")


from wordcloud import WordCloud, STOPWORDS
STOPWORDS.add('breast')
STOPWORDS.add('cancer')
plt.figure(figsize=(8,6))
subset = train_df[train_df.Narrative==1]
text = subset.Message.values
cloud_toxic = WordCloud(stopwords=STOPWORDS, background_color='white', collocations=False, width=1920, height=1080).generate(" ".join(text))
plt.axis('off')
plt.title("Narrative",fontsize=20)
plt.imshow(cloud_toxic)

from wordcloud import WordCloud, STOPWORDS
STOPWORDS.add('breast')
STOPWORDS.add('cancer')
plt.figure(figsize=(8,6))
subset = train_df[train_df.Narrative==0]
text = subset.Message.values
cloud_toxic = WordCloud(stopwords=STOPWORDS, background_color='white', collocations=False, width=1920, height=1080).generate(" ".join(text))
#cloud_toxic = WordCloud(background_color='white', collocations=False, width=1920, height=1080).generate(" ".join(text))
plt.axis('off')
plt.title("Non-narrative",fontsize=20)
plt.imshow(cloud_toxic)

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



stop_words = STOPWORDS
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

print(stop_words)

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

stopwords= set(["couldn't", 'these', "they're", "won't", 'some', 'two', "it's", 'their', 'because', 'them', 'day', 'ought', "we're", 'on', "what's", 'where', 'by', 'beside', 'yourselves', 'did', 's', 'out', "when's", 'however', 'like', 'at', 'could', 'and', 'how', "they'll", 'among', 'here', 'six', 'before', 'is', 'too', "we'll", 'there', 'as', 'breast', "i'll", "haven't", "mustn't", "can't", 'each', 'her', 'can', 'between', 'few', "doesn't", 'cannot', 'does', "that's", 'since', 'ten', 'only', "you've", 'until', 'itself', "i'd", 'now', 'the', "shan't", 'for', 'into', 'also', 'through', "we've", 'above', 'if', 'or', 'ours', "hadn't", "where's", "didn't", 'doing', 'most', 'otherwise', 'that', 'when', 'just', 'from', 'ever', "he's", 'should', 'herself', "you'd", 'com', 'its', "they'd", 'this', 'than', 'up', 'have', 'again', 'been', 'him', "they've", 'but', 'he', 'me', 'no', 'to', "hasn't", "don't", 'his', 'nine', 'do', "here's", 'which', 'being', 'those', 'himself', 'both', 'down', 'has', 'such', 'five', 'i', "wasn't", 'three', 'own', 'may', 'other', "he'd", 'themselves', 'having', 'be', "i'm", 'www', "you're", "she'll", 'it', "wouldn't", 'eight', 'not', 'were', 'cancer', 'we', 'an', 'k', 'hers', "i've", "there's", 'yours', 'any', 'get', 'over', 'what', 'one', 'our', 'why', 'yourself', 'while', 'will', 'once', 'they', 'against', 'theirs', 'four', 'very', 'across', 'so', 'about', "he'll", 'further', 'within', 'under', 'whom', 'same', 'a', 'more', 'during', 'my', 'ourselves', 'she', 'off', 'am', 'would', 'all', 'then', 'you', 'had', "aren't", "why's", 'zero', 't', "weren't", 'http', 'r', 'in', "how's", 'who', "who's", 'shall', 'after', 'else', "she's", 'was', 'your', 'seven', "you'll", "isn't", 'don', 'nor', "she'd", "shouldn't", "let's", 'of', "we'd", 'with', 'below', 'are', 'myself'])

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
#model = LogisticRegression(C=12)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

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
np.savetxt("result_RandomForest.csv", preds_test, delimiter=",")
files.download('result_RandomForest.csv')
