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
import collections
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from wordcloud import WordCloud, STOPWORDS

from google.colab import files
uploaded = files.upload()
train_df =pd.read_csv('train.csv',error_bad_lines=False, engine="python")

#from wordcloud import WordCloud, STOPWORDS
from wordcloud import WordCloud
STOPWORDS = {''}
STOPWORDS.add('breast')
STOPWORDS.add('cancer')
plt.figure(figsize=(8,6))
subset = train_df[train_df.Narrative==1]
text = subset.Message.values
cloud_toxic = WordCloud(stopwords=STOPWORDS, background_color='white', collocations=False, width=1920, height=1080).generate(" ".join(text))
#cloud_toxic = WordCloud(background_color='white', collocations=False, width=1920, height=1080).generate(" ".join(text))
plt.axis('off')
plt.title("Narrative",fontsize=20)
plt.imshow(cloud_toxic)

sum =''
for i in range(len(text)):
  sum  = sum+ text[i]
sum=sum.lower()

filtered_words=[]
for word in sum.split():
  filtered_words.append(word)
  
counted_words = collections.Counter(filtered_words)

words1 = []
counts1 = []
for letter, count in counted_words.most_common(100):
    words1.append(letter)
    counts1.append(count)

#from wordcloud import WordCloud, STOPWORDS
from wordcloud import WordCloud
STOPWORDS = {''}
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

sum =''
for i in range(len(text)):
  sum  = sum+ text[i]

sum=sum.lower()

filtered_words=[]
for word in sum.split():
  filtered_words.append(word)

counted_words = collections.Counter(filtered_words)

words2 = []
counts2 = []
for letter, count in counted_words.most_common(100):
    words2.append(letter)
    counts2.append(count)

non_unique_word_2 =[]
non_unique_word_1=[]
unique_word_2 =[]
unique_word_1=[]

for i in range(100):
  if((words2[i] in words1)== True):
    non_unique_word_2.append(words2[i])

for i in range(100):
  if((words1[i] in words2)== True):
    non_unique_word_1.append(words1[i])

for i in range(100):
  if((words2[i] in words1)== False):
    unique_word_2.append(words2[i])

for i in range(100):
  if((words1[i] in words2)== False):
    unique_word_1.append(words1[i])
    
print(non_unique_word_2)
print(non_unique_word_1)
print(unique_word_2)
print(unique_word_1)
