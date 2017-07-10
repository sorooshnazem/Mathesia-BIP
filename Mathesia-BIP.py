print(__doc__)
import os.path
import csv
import os
import pandas as pd
from sklearn import tree 
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_curve, f1_score



url = "/Users/sorooshnazem/Downloads/Dataset/train.csv"
clients = pd.read_csv(url)
clientsInfo=pd.DataFrame()
columns=(str(clients.columns.values).split(','))
for value in columns:
    value=value.translate(None, "[]'\" ")
    clientsInfo[value] = 0

for i in range(0,len(clients.as_matrix())):
    #for i in range(0,100):
    recordValues=(str(clients.as_matrix()[i])).split(',')
    recordValues=[value.translate(None, "[]'\" ") for value in recordValues]
    clientsInfo.loc[i]=recordValues

clientsInfo['CUST_COD'] = clientsInfo['CUST_COD'].astype(int)
clientsInfo['LIMIT_BAL'] = clientsInfo['LIMIT_BAL'].astype(float)
clientsInfo['BILL_AMT_DEC']=clientsInfo['BILL_AMT_DEC'].astype(float)
clientsInfo['BILL_AMT_NOV']=clientsInfo['BILL_AMT_NOV'].astype(float)
clientsInfo['BILL_AMT_OCT']=clientsInfo['BILL_AMT_OCT'].astype(float)
clientsInfo['BILL_AMT_SEP']=clientsInfo['BILL_AMT_SEP'].astype(float)
clientsInfo['BILL_AMT_AUG']=clientsInfo['BILL_AMT_AUG'].astype(float)
clientsInfo['BILL_AMT_JUL']=clientsInfo['BILL_AMT_JUL'].astype(float)
clientsInfo['PAY_AMT_DEC']=clientsInfo['PAY_AMT_DEC'].astype(float)
clientsInfo['PAY_AMT_NOV']=clientsInfo['PAY_AMT_NOV'].astype(float)
clientsInfo['PAY_AMT_OCT']=clientsInfo['PAY_AMT_OCT'].astype(float)
clientsInfo['PAY_AMT_SEP']=clientsInfo['PAY_AMT_SEP'].astype(float)
clientsInfo['PAY_AMT_AUG']=clientsInfo['PAY_AMT_AUG'].astype(float)
clientsInfo['PAY_AMT_JUL']=clientsInfo['PAY_AMT_JUL'].astype(float)
clientsInfo = clientsInfo[clientsInfo.BIRTH_DATE != '#N/A']
clientsInfo = clientsInfo[clientsInfo.MARRIAGE != '#N/A']

clientsInfo['BIRTH_DATE'] = pd.to_datetime(clientsInfo['BIRTH_DATE'])

clientsInfo['PAY_DEC']=clientsInfo['PAY_DEC'].astype(int)
clientsInfo['PAY_NOV']=clientsInfo['PAY_NOV'].astype(int)
clientsInfo['PAY_OCT']=clientsInfo['PAY_OCT'].astype(int)
clientsInfo['PAY_SEP']=clientsInfo['PAY_SEP'].astype(int)
clientsInfo['PAY_AUG']=clientsInfo['PAY_AUG'].astype(int)
clientsInfo['PAY_JUL']=clientsInfo['PAY_JUL'].astype(int)

target = clientsInfo['DEFAULTPAYMENTJAN'].values


clientsInfo['MARRIAGE'] = np.where(clientsInfo['MARRIAGE']== 'single', 0,1)
clientsInfo['EDUCATION'] = np.where(clientsInfo['EDUCATION']== 'university', 0,np.where(clientsInfo['EDUCATION']== 'highschool',1,2))

features = clientsInfo[['CUST_COD','PAY_DEC', 'PAY_NOV', 'PAY_OCT', 'PAY_SEP',
                        'PAY_AUG', 'PAY_JUL', 'BILL_AMT_DEC', 'BILL_AMT_NOV',
                        'BILL_AMT_OCT', 'BILL_AMT_SEP', 'BILL_AMT_AUG', 'BILL_AMT_JUL',
                        'PAY_AMT_DEC', 'PAY_AMT_NOV', 'PAY_AMT_OCT', 'PAY_AMT_SEP',
                        'PAY_AMT_AUG', 'PAY_AMT_JUL','MARRIAGE','EDUCATION']].values

my_tree = tree.DecisionTreeClassifier()
my_tree = my_tree.fit(features, target)

#Here the results of prediction are shown
print(my_tree.feature_importances_)
print(my_tree.score(features, target))

my_prediction = my_tree.predict(features)
print accuracy_score(target, my_prediction)
print precision_recall_curve(target, my_prediction)
print f1_score(target, my_prediction)

#Test Data Prediction

url = "/Users/sorooshnazem/Downloads/Dataset/test.csv"
clients_test = pd.read_csv(url)
clientsInfo_test=pd.DataFrame()
columns=(str(clients_test.columns.values).split(';'))
for value in columns:
    value=value.translate(None, "[]'\" ")
    clientsInfo_test[value] =0

for i in range(0,len(clients_test.as_matrix())):
    #for i in range(0,100):
    recordValues=(str(clients_test.as_matrix()[i])).split(';')
    recordValues=[value.translate(None, "[]'\" ") for value in recordValues]
    clientsInfo_test.loc[i]=recordValues

clientsInfo_test['CUST_COD'] = clientsInfo_test['CUST_COD'].astype(int)
clientsInfo_test['LIMIT_BAL'] = clientsInfo_test['LIMIT_BAL'].astype(float)
clientsInfo_test['BILL_AMT_DEC']=clientsInfo_test['BILL_AMT_DEC'].astype(float)
clientsInfo_test['BILL_AMT_NOV']=clientsInfo_test['BILL_AMT_NOV'].astype(float)
clientsInfo_test['BILL_AMT_OCT']=clientsInfo_test['BILL_AMT_OCT'].astype(float)
clientsInfo_test['BILL_AMT_SEP']=clientsInfo_test['BILL_AMT_SEP'].astype(float)
clientsInfo_test['BILL_AMT_AUG']=clientsInfo_test['BILL_AMT_AUG'].astype(float)
clientsInfo_test['BILL_AMT_JUL']=clientsInfo_test['BILL_AMT_JUL'].astype(float)
clientsInfo_test['PAY_AMT_DEC']=clientsInfo_test['PAY_AMT_DEC'].astype(float)
clientsInfo_test['PAY_AMT_NOV']=clientsInfo_test['PAY_AMT_NOV'].astype(float)
clientsInfo_test['PAY_AMT_OCT']=clientsInfo_test['PAY_AMT_OCT'].astype(float)
clientsInfo_test['PAY_AMT_SEP']=clientsInfo_test['PAY_AMT_SEP'].astype(float)
clientsInfo_test['PAY_AMT_AUG']=clientsInfo_test['PAY_AMT_AUG'].astype(float)
clientsInfo_test['PAY_AMT_JUL']=clientsInfo_test['PAY_AMT_JUL'].astype(float)
clientsInfo_test = clientsInfo_test[clientsInfo_test.BIRTH_DATE != '#N/A']
clientsInfo_test = clientsInfo_test[clientsInfo_test.MARRIAGE != '#N/A']

clientsInfo_test['PAY_DEC']=clientsInfo_test['PAY_DEC'].astype(int)
clientsInfo_test['PAY_NOV']=clientsInfo_test['PAY_NOV'].astype(int)
clientsInfo_test['PAY_OCT']=clientsInfo_test['PAY_OCT'].astype(int)
clientsInfo_test['PAY_SEP']=clientsInfo_test['PAY_SEP'].astype(int)
clientsInfo_test['PAY_AUG']=clientsInfo_test['PAY_AUG'].astype(int)
clientsInfo_test['PAY_JUL']=clientsInfo_test['PAY_JUL'].astype(int)
clientsInfo_test['MARRIAGE'] = np.where(clientsInfo_test['MARRIAGE']== 'single', 0,1)
clientsInfo_test['EDUCATION'] = np.where(clientsInfo_test['EDUCATION']== 'university', 0,np.where(clientsInfo_test['EDUCATION']== 'highschool',1,2))

features_test = clientsInfo_test[['CUST_COD','PAY_DEC', 'PAY_NOV', 'PAY_OCT', 'PAY_SEP',
                                  'PAY_AUG', 'PAY_JUL', 'BILL_AMT_DEC', 'BILL_AMT_NOV',
                                  'BILL_AMT_OCT', 'BILL_AMT_SEP', 'BILL_AMT_AUG', 'BILL_AMT_JUL',
                                  'PAY_AMT_DEC', 'PAY_AMT_NOV', 'PAY_AMT_OCT', 'PAY_AMT_SEP',
                                  'PAY_AMT_AUG', 'PAY_AMT_JUL','MARRIAGE','EDUCATION']].values
target_test = clientsInfo_test[['DEFAULTPAYMENTJAN']].values

my_tree = tree.DecisionTreeClassifier()
my_tree = my_tree.fit(features,target)
paymentDefaultPrediction=my_tree.predict(features_test)

clientsInfo_test['DEFAULTPAYMENTJAN']=paymentDefaultPrediction

url = "/Users/sorooshnazem/Downloads/Dataset/test_done.csv"
clientsInfo_test.to_csv(url)

