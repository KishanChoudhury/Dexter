import re
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans


def ParseText(Str):
    
    #text="".join([ch for ch in Str if ch not in string.punctuation])
    text_list=Str.split(" ")
    stemmer=SnowballStemmer("english")
    stemmed_list=[]
    sw=stopwords.words("english")
    for word in text_list:
        stripped_word=word.strip()
        if stemmer.stem(stripped_word) not in sw:
            stemmed_list.append(stemmer.stem(stripped_word))
    return " ".join(stemmed_list)
    

def preprocess(text):
    text=re.sub('<[^>]+>', '', text)
    text=text.replace('A:','')
    text=text.replace('B:','')
    text=text.replace('RER','')
    #text5=text4.translate(None, string.punctuation)
    text="".join(l for l in text if l not in string.punctuation)
    cleaned_text=" ".join(text.split())
    return cleaned_text

#load training data
import glob
path='F:/BacktoRobotics/Hackathon/Data/CCCS-Decoda-FR-EN-training_2015-01-30/CCCS-Decoda-FR-EN-training_2015-01-30/EN/auto_no_synopsis/temp/*.txt'
files=glob.glob(path)
traindata={'Conversation':{},'Labels':{}}
i=0
for file in files:
    with open(file) as f:
        text=f.read()
        processed_text=preprocess(text)
        stemmed_text=ParseText(processed_text)
        i+=1
        traindata['Conversation'][i]=stemmed_text

i=0
import csv
LabelFilePath='F:\BacktoRobotics\Hackathon\Category.csv'
with open(LabelFilePath,'r') as f:
    reader=csv.reader(f)
    for row in reader:
        i+=1
        traindata['Labels'][i]=row[1]

df=pd.DataFrame(traindata)

features=df['Conversation']
labels=df['Labels']



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(features,labels,test_size=.3,random_state=42)


vec=CountVectorizer()
vec.fit(df['Conversation'].values.astype('U'))
X_train_transformed=vec.transform(X_train.values.astype('U'))
X_test_transformed=vec.transform(X_test.values.astype('U'))



from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(X_train_transformed.toarray(),Y_train)
Y_pred=clf.predict(X_test_transformed.toarray())
print(accuracy_score(Y_test,Y_pred))
