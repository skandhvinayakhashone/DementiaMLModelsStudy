
# coding: utf-8


from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

import pandas as pd
import sklearn.metrics 
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import numpy as np

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



#encoding the levels in y
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(features)

#importing keras and converting y to catagorical
import keras
from keras.utils import np_utils
y = np_utils.to_categorical(y)

#splitting dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pd_data, y, test_size = 0.25, random_state = 20000)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Import Keras libraries 
from keras.models import Sequential
from keras.layers import Dense



# ANN
classifier = Sequential()

classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 7))

classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 5, epochs = 60) # Lesser no of epochs - Basic Model

# Prediction
y_pred = classifier.predict(X_test)

maxi = y_pred.max(axis=1)


for i in range(len(y_pred)):
    for j in range(3):
        if y_pred[i,j] == maxi[i]:
           y_pred[i,j] = 1
        else:
               y_pred[i,j] = 0
     

# Accuracy    
crt_values = (y_pred == y_test).sum()
wrong_values = (y_pred != y_test).sum()
total = crt_values+wrong_values
result = crt_values/total
print(result) 

#92.9 percent accuracy

