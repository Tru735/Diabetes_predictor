import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

df = pd.read_csv('C:/Users/tjsss/OneDrive/Desktop/machine learning/diabetes project/dataset/diabetes.csv')

df['Outcome'].value_counts()

Y = df['Outcome']
X = df.drop(columns = 'Outcome' , axis = 1)

scaler= StandardScaler()
standard_data = scaler.fit_transform(X)

X = standard_data

X_train , X_test , Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify = Y, random_state = 1234)

print(X_train.shape, Y_train.shape)

classifier = svm.SVC(kernel = 'linear')
classifier.fit(X_train , Y_train)
train_prediction = classifier.predict(X_train)
train_accuracy = accuracy_score(train_prediction , Y_train)
print(train_accuracy)

print("************************************************test accuracy************************************************")
test_prediction = classifier.predict(X_test)
test_accuracy = accuracy_score(test_prediction , Y_test)
print( test_accuracy)

input_data  =(3,171,72,33,135,33.3,0.199,24)
input_data_np = np.asarray(input_data)
reshaped_input = input_data_np.reshape(1,-1)

std_data = scaler.transform(reshaped_input)

classifier.predict(std_data)
print(classifier.predict(std_data))

if(classifier.predict(std_data)[0]==0):
    print("not diabetic")
else:
    print("diabetic")