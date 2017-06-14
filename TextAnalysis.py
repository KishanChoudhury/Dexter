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
path='F:/BacktoRobotics/Hackathon/Data/CCCS-Decoda-FR-EN-training_2015-01-30/CCCS-Decoda-FR-EN-training_2015-01-30/EN/auto/text/*.txt'
files=glob.glob(path)
traindata={'Conversation':{}}
i=0;
for file in files:
    with open(file) as f:
        text=f.read()
        processed_text=preprocess(text)
        stemmed_text=ParseText(processed_text)
        i+=1
        traindata['Conversation'][i]=stemmed_text

df=pd.DataFrame(traindata)

vec=CountVectorizer()
vec.fit(df['Conversation'])
X=vec.transform(df['Conversation'])

kmeans=KMeans(n_clusters=2,random_state=0).fit(X)
print(kmeans.labels_)


