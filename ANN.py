ANN

!pip install tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf

dataset=pd.read_csv("D:\MBA - Balaji\Sem 3\Deep Learning\ANN_Modelling.csv")#excel file location
X = dataset.iloc[:,3:-1].values
Y = dataset.iloc[:,-1].values
dataset

#Label encoding the "Gender" column

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])
X

#One hot encoding the "Geography" column

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X

#Splitting the data

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.25,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Initializaing the ANN
ann=tf.keras.models.Sequential()

#Adding the input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#Adding the input layer and second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#Compiling the ANN
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Training the ANN on the training set
ann.fit(X_train, Y_train, batch_size = 32, epochs=100)


#Predicting the output

Y_pred=ann.predict(X_test)
Y_pred=(Y_pred > 0.5)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1),Y_test.reshape(len(Y_test),1)),1))


#Making confusion matrix for accuracy

from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(Y_test,Y_pred)
print(cm)
accuracy_score(Y_test,Y_pred)