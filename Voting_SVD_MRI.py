
# coding: utf-8

from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

import sklearn.metrics 
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

X = pd.read_csv("/Users/skandhvinayak/Downloads/mri-and-alzheimers/oasis_longitudinal.csv",sep=",")#get data set
features=X[['Group']]#seperate features
data=X[['Visit' , 'MR Delay', 'M/F', 'Hand',  'Age', 'EDUC',  'SES',  'MMSE',  'CDR',  'eTIV','nWBV','ASF']]#seperate the data columns
data['M/F'] = pd.factorize(data['M/F'])[0]
data['Hand'] = pd.factorize(data['Hand'])[0]



data.fillna(data.mean(), inplace=True)#fill na with mean
#print(data.isnull().values.ravel().sum())
x = data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()#get the scaler function
x_scaled = min_max_scaler.fit_transform(x)#scale ie normalize the columns
data = pd.DataFrame(x_scaled)#make the normalized values into a pandas data set
svd = TruncatedSVD(n_components=7)#choose the number of componenets in the final dataset
new_data=svd.fit_transform(data)#fit the svd values to the dataset and store it in a new dataset
pd_data=pd.DataFrame(new_data)#convert new_data into pandas
X_train, X_test, y_train, y_test = train_test_split(pd_data, features, test_size = 0.25, random_state =2)#split the dataset

#create fucntion alaises
knc=KNeighborsClassifier(n_neighbors=19)
dtc = DecisionTreeClassifier()
rfc=RandomForestClassifier(random_state=226)
bc=BaggingClassifier(DecisionTreeClassifier(),n_estimators=9)
adc=AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 5, learning_rate = 1)
lr=LogisticRegression()

vc = VotingClassifier( estimators= [('lr',lr),('dtc',dtc),('knc',knc),('rfc',rfc),('bc',bc),('adc',adc)], voting = 'hard')#get the VotingClassifier Function

vc.fit(X_train,y_train.values.ravel())#create model using train set

y_pred = vc.predict(X_test)#predict the values
cm=confusion_matrix(y_test,y_pred)#get the confusion matrix
diag_sum=0
for i in range(0,3):
    diag_sum=diag_sum+cm[i,i]
total=0
for i in range(0,3):
    for j in range(0,3):
        total=total+cm[i,j]
accuracy=diag_sum/total# accuracy of the confusion matrix
#get accuracy and confusion matrix
print(accuracy)
print(cm)

#93.6 percent accuracy
