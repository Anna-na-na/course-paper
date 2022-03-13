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

# Каталог, в котором находится распакованный набор данных
basepath = 'nlp-getting-started'

train_data = pd.read_csv(os.path.join(basepath, 'train.csv'))
print(train_data.head())

## CLEAN
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

train_data = standardize_text(train_data, "text")

stop = stopwords.words('english')
pat = r'\b(?:{})\b'.format('|'.join(stop))
train_data["text"] = train_data["text"].str.replace(pat, '')

print(train_data.head())

## STEMMING
porter = PorterStemmer()

def tokenizer_porter(text):
    return " ".join(porter.stem(word) for word in text.split())

tokens = np.array(train_data["text"].apply(tokenizer_porter))

print(tokens[:5])


## BAG OF WORDS
count = CountVectorizer()
bag = count.fit_transform(tokens)
print(bag.toarray()[:5])


## GRAPHICS
def plot_LSA(test_data, test_labels, plot=True):
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    # color_mapper = {label: idx for idx, label in enumerate(set(test_labels))}
    # color_column = [color_mapper[label] for label in test_labels]
    colors = ['maroon', 'lightseagreen']
    if plot:
        plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=8, alpha=.8, c=test_labels,
                    cmap=matplotlib.colors.ListedColormap(colors))
        disaster_patch = mpatches.Patch(color='maroon', label='Disaster tweets')
        ord_patch = mpatches.Patch(color='lightseagreen', label='Ordinary tweets')
        plt.legend(handles=[ord_patch, disaster_patch])

labels = train_data["target"].tolist()
fig = plt.figure(figsize=(10,10))
plot_LSA(bag, labels)
plt.show()

## TF-IDF
tfidf = TfidfVectorizer()
new_bag = tfidf.fit_transform(tokens)
print(new_bag.toarray()[:5])

fig = plt.figure(figsize=(10,10))
plot_LSA(new_bag, labels)
plt.show()