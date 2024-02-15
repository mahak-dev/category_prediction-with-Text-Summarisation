try:
    import os

    import pandas as  pd
    import spacy

    import seaborn as sns
    import string

    from tqdm import tqdm
    from textblob import TextBlob

    from nltk.corpus import stopwords
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk import word_tokenize
    import re


    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline


    from sklearn.preprocessing import FunctionTransformer
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import FeatureUnion
    from sklearn.feature_extraction import DictVectorizer


    tqdm.pandas()
except Exception as e:
    print("Error : {} ".format(e))

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

df = pd.read_csv("test.csv")

sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')

df['Category'].value_counts().plot( kind='bar', figsize=(15,10))

df.columns

df.describe()

df.isna().sum()

df.dropna(inplace=True)

df.isna().sum()

df.head(2)

df['Category'].unique()

stop_words_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()
my_sw = ['make', 'amp',  'news','new' ,'time', 'u','s', 'photos',  'get', 'say']

def black_txt(token):
    return  token not in stop_words_ and token not in list(string.punctuation)  and len(token)>2 and token not in my_sw

def clean_txt(text):
    clean_text = []
    clean_text2 = []
    text = re.sub("'", "",text)
    text=re.sub("(\\d|\\W)+"," ",text)
    clean_text = [ wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if black_txt(word)]
    clean_text2 = [word for word in clean_text if black_txt(word)]
    return " ".join(clean_text2)

def subj_txt(text):
    return  TextBlob(text).sentiment[1]

def polarity_txt(text):
    return TextBlob(text).sentiment[0]

def len_text(text):
    if len(text.split())>0:
         return len(set(clean_txt(text).split()))/ len(text.split())
    else:
         return 0

df['text'] = df['Headline']  +  " " + df['Summary']

df['text'] = df['text'].swifter.apply(clean_txt)
df['polarity'] = df['text'].swifter.apply(polarity_txt)
df['subjectivity'] = df['text'].swifter.apply(subj_txt)
df['len'] = df['text'].swifter.apply(lambda x: len(x))

!pip install swifter

import swifter

print(df['text'].dtype)

df['text'] = df['text'].astype(str)

def clean_txt(text):
    # ...
    text = str(text)
    text = re.sub("'", "", text)
    # ...



df['text'] = df['Headline']  +  " " + df['Summary']
df['text'] = df['text'].apply(lambda x : str(x))

df['polarity'] = df['text'].swifter.apply(polarity_txt)
df['subjectivity'] = df['text'].swifter.apply(subj_txt)
df['len'] = df['text'].swifter.apply(lambda x: len(x))

print(df['text'].dtype)

df['text'] = df['text'].astype(str)

X = df[['text', 'polarity', 'subjectivity','len']]
y =df['Category']

encoder = LabelEncoder()
y = encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
v = dict(zip(list(y), df['Category'].to_list()))

text_clf = Pipeline([
...     ('vect', CountVectorizer(analyzer="word", stop_words="english")),
...     ('tfidf', TfidfTransformer(use_idf=True)),
...     ('clf', MultinomialNB(alpha=.01)),
... ])

text_clf.fit(x_train['text'].to_list(), list(y_train))

import numpy as np

X_TEST = x_test['text'].to_list()
Y_TEST = list(y_test)

predicted = text_clf.predict(X_TEST)

c = 0

for doc, category in zip(X_TEST, predicted):

    if c == 2:break

    print("-"*55)
    print(doc)
    print(v[category])
    print("-"*55)

    c = c + 1

Accuracy = np.mean(predicted == Y_TEST)
Accuracy

docs_new = ["He was subdued by passengers and crew when he fled to the back of the aircraft after the confrontation, according to the U.S. attorney's office in Los Angeles."]

predicted = text_clf.predict(docs_new)

v[predicted[0]]

import pickle
with open('model.pkl','wb') as f:
    pickle.dump(text_clf,f)

with open('model.pkl', 'rb') as f:
    clf2 = pickle.load(f)

df2 = pd.read_csv('test.csv')

df2.dropna(inplace = True)

docs_new = df2['Summary']
predicted = clf2.predict(docs_new)

docs_new = ["He was subdued by passengers and crew when he fled to the back of the aircraft after the confrontation, according to the U.S. attorney's office in Los Angeles."]
predicted = clf2.predict(docs_new)

v[predicted[0]]

