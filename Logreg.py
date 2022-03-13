import pandas as pd
import numpy as np
import os
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Каталог, в котором находится распакованный набор данных
basepath = 'nlp-getting-started'

train_data = pd.read_csv(os.path.join(basepath, 'train.csv'))
test_data = pd.read_csv(os.path.join(basepath, 'test.csv'))
test_submission = pd.read_csv(os.path.join(basepath, 'sample_submission.csv'))

def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

train_data = standardize_text(train_data, "text")
test_data = standardize_text(test_data, "text")

stop = stopwords.words('english')

def tokenizer(text):
    return text.split()

def stop_tokenizer(text):
    pat = r'\b(?:{})\b'.format('|'.join(stop))
    text = text.replace(pat, '')
    return text.split()

porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

def stop_tokenizer_porter(text):
    pat = r'\b(?:{})\b'.format('|'.join(stop))
    text = text.replace(pat, '')
    return [porter.stem(word) for word in text.split()]

X_train = train_data.loc[:, 'text'].values
y_train = train_data.loc[:, 'target'].values
X_test = test_data.loc[:, 'text'].values
y_test = test_submission.loc[:, 'target'].values

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)
param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__tokenizer': [tokenizer, tokenizer_porter,
                                   stop_tokenizer, stop_tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__tokenizer': [tokenizer, tokenizer_porter,
                                   stop_tokenizer, stop_tokenizer_porter],
               'vect__use_idf': [False],
               'vect__norm': [None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]}]
lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf',
                      LogisticRegression(random_state=0,
                                         solver='liblinear', max_iter=1000))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5, verbose=2,
                           n_jobs=1)
gs_lr_tfidf.fit(X_train, y_train)
print('Наилучший набор параметров: ')
for keys, values in gs_lr_tfidf.best_params_.items():
    print(keys, ' ', values)

print('Правильность при перекрестной проверке на тренировочной выборке: %.3f'
      % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Правильность при испытании на тестовой выборке: %.3f'
      % clf.score(X_test, y_test))