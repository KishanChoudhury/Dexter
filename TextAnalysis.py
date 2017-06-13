import re
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
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
traindata={}
i=0;
for file in files:
    with open(files[0]) as f:
        text=f.read()
        processed_text=preprocess(text)
        i+=1
        traindata[i]={}
        traindata[i]['Conversation']=processed_text


df=pd.DataFrame(traindata)
print(df.describe())        



