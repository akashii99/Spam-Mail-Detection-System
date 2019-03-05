#import all the needed libraries
import mailbox
import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas as pd
#import cPickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
#from sklearn.pipeline import Pipeline
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier 
#from sklearn.learning_curve import learning_curve

#import metrics libraries
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

#function to get email text from email body
def getmailtext(message): #getting plain text 'email body'
    body = None
    #check if mbox email message has multiple parts
    if message.is_multipart():
        for part in message.walk():
            if part.is_multipart():
                for subpart in part.walk():
                    if subpart.get_content_type() == 'text/plain':
                        body = subpart.get_payload(decode=True)
            elif part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True)
    #if message only has a single part            
    elif message.get_content_type() == 'text/plain':
        body = message.get_payload(decode=True)
    #return mail text which concatenates both mail subject and body
    mailtext=str(message['subject'])+" "+str(body)
    return mailtext


#read spam mbox email file
mbox = mailbox.mbox('Spam.mbox')

mlist_spam = []
#create list which contains mail text for each spam email message
for message in mbox:
    mlist_spam.append(getmailtext(message))
    #break
#read ham mbox email file
mbox_ham = mailbox.mbox('ham.mbox')

mlist_ham = []
count=0
#create list which contains mail text for each ham email message
for message in mbox_ham:
    
    mlist_ham.append(getmailtext(message))
    if count>601:
        break
    count+=1

#create 2 dataframes for ham spam mails which contain the following info-
#Mail text, mail length, mail is ham/spam label
spam_df = pd.DataFrame(mlist_spam, columns=["message"])
spam_df["label"] = "spam"

spam_df['length'] = spam_df['message'].map(lambda text: len(text))
print(spam_df.head())

ham_df = pd.DataFrame(mlist_ham, columns=["message"])
ham_df["label"] = "ham"

ham_df['length'] = ham_df['message'].map(lambda text: len(text))
print(ham_df.head())