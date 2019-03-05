import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import re
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

raw_data = pd.read_csv("SMSSpamCollection.tsv", sep='\t', names=['label','text'])
pd.set_option('display.max_colwidth',100)
print(raw_data.head())
print(raw_data.shape)
print(raw_data['label'].value_counts())
#pd.crosstab(raw_data['label'],columns = 'label',normalize=True)

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

def punct_pc(text):
    punct_count = sum([1 for char in text if char in string.punctuation])
    return (punct_count/(len(text) - text.count(' ')))*100

raw_data['text_length'] = raw_data['text'].apply(lambda x : len(x)-x.count(' '))
raw_data['Punct_pc'] = raw_data['text'].apply(lambda x : punct_pc(x))

def clean_data(text):
    punct = "".join([word.lower() for word in text if word not in string.punctuation])
    splt = re.split('\W+',punct)
    txt = [ps.stem(word) for word in splt if word not in stopwords]
    return txt

#print(raw_data.head())

X_train,X_test,Y_train,Y_test = train_test_split(raw_data[['text','text_length','Punct_pc']],raw_data['label'],test_size=0.2,random_state=123)

'''print(pd.crosstab(Y_train,columns = 'label',normalize=True))
print(pd.crosstab(Y_test,columns = 'label',normalize=True))
print(X_train.head())'''

Tfidf_Vect = TfidfVectorizer(analyzer=clean_data)
Tfidf_vect_fit = Tfidf_Vect.fit(X_train['text'])

X_train_Tfidf_vect = Tfidf_vect_fit.transform(X_train['text'])
X_test_Tfidf_vect = Tfidf_vect_fit.transform(X_test['text'])

X_train_vect = pd.concat([X_train[['text_length','Punct_pc']].reset_index(drop=True) ,
                         pd.DataFrame(X_train_Tfidf_vect.toarray())],axis=1)


X_test_vect = pd.concat([X_test[['text_length','Punct_pc']].reset_index(drop=True) , 
                        pd.DataFrame(X_test_Tfidf_vect.toarray())],axis=1)
#print(X_train_vect.head())


#RandomForest Classifier
'''rf = RandomForestClassifier(random_state=123,n_jobs=3)
param = {'n_estimators' : [10,25,50,100,300], 'max_depth' : [10, 20, 50,100, None],'max_features' : [10,50,'auto']}

grid = GridSearchCV(rf,param,cv=5,n_jobs=3)

rf_grid_fit_1 = grid.fit(X_train_vect, Y_train)
df3 = print(pd.DataFrame(rf_grid_fit_1.cv_results_).sort_values('mean_test_score',ascending=False,return_train_score=True))
print(df3.head())

RF_results_TFIDF = pd.DataFrame(rf_grid_fit_1.cv_results_).sort_values('mean_test_score',ascending=False)
RF_results_TFIDF.to_csv("result_tfidf.csv",header=True)'''

import time

rf_final_1 = RandomForestClassifier(n_estimators = 50, max_depth = 100, max_features='auto',n_jobs=-1,random_state=123)

start = time.time()
rf_model_1 = rf_final_1.fit(X_train_vect, Y_train)
end = time.time()
fit_time = end - start

start = time.time()
Y_pred = rf_model_1.predict(X_test_vect)
end = time.time()
predict_time = end-start

precision, recall, fscore, train_support = score(Y_test, Y_pred, pos_label='spam', average='binary')
print('Fit_time : {} / Predict_time : {} / Precision: {} / Recall: {} / Accuracy: {}'.format(round(fit_time,3),round(predict_time,3),round(precision, 3), round(recall, 3), round((Y_pred==Y_test).sum()/len(Y_pred), 3)))
