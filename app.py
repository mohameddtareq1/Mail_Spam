import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn import naive_bayes
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import string
import re
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')


df = pd.read_csv("/content/data_spam.csv",encoding='latin-1')
df.head()

df.columns
df.drop("Unnamed: 2",inplace=True,axis=1)
df.drop("Unnamed: 3",inplace=True,axis=1)
df.drop("Unnamed: 4",inplace=True,axis=1)


df.head()

df.isna().sum()

df.info()

# 1-> ham
# 0-> spam
df.loc[df["v1"]=="spam","v1"] = 0
df.loc[df["v1"]=="ham","v1"] = 1

df.head()


stop_word=set(stopwords.words("english"))
df["v2"]=df["v2"].apply(nltk.word_tokenize)
df["v2"] = [[word for word in sentence if not word in stop_word and word.isalnum()] for sentence in df["v2"]]


word_lemmatizer = WordNetLemmatizer()
df["v2"]=df["v2"].apply(lambda x: [word_lemmatizer.lemmatize(word) for word in x])


df.head()

df['v2'] = df['v2'].apply(lambda x: ' '.join(x))


df.head()

X=df["v2"]
Y=df["v1"]



print(X)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

print(X.shape,X_train.shape,X_test.shape)

print(Y.shape,Y_train.shape,Y_test.shape)

feature_extraction = TfidfVectorizer(min_df=1,stop_words="english",lowercase=True)
X_train_feature = feature_extraction.fit_transform(X_train)
X_test_feature = feature_extraction.transform(X_test)

Y_train=Y_train.astype("int")
Y_test=Y_test.astype("int")
X_train.shape

Model= LogisticRegression()

Model.fit(X_train_feature,Y_train)

prediction_train = Model.predict(X_train_feature)
cm_train=confusion_matrix(Y_train,prediction_train)
print(cm_train)

print(classification_report(Y_train,prediction_train))



prediction_test= Model.predict(X_test_feature)
cm_test=confusion_matrix(Y_test,prediction_test)
print(cm_test)

print(classification_report(Y_test,prediction_test))

Nb_model = naive_bayes.MultinomialNB()
Nb_model.fit(X_train_feature,Y_train)


prediction_Nb = Nb_model.predict(X_test_feature)
cm_Nb=confusion_matrix(Y_test,prediction_Nb)
print(cm_Nb)

print(classification_report(Y_test,prediction_Nb))
