from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

#change the url only

url='https://raw.githubusercontent.com/sak1b0/Thesis/master/abalone.csv'

data=pd.read_csv(url,header=None)
data=data.strip(' ')
data=np.asarray(data)

X = np.delete(data, data.shape[1] - 1, axis=1)
y = data[:, -1]

print(X.shape)
print(y.shape)

'''
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3,random_state=109) # 70% training and 30% test

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

model = GaussianNB()
model.fit(dataset.data, dataset.target)
expected = dataset.target
predicted = model.predict(dataset.data)
print(metrics.classification_report(expected, predicted))
#print(metrics.confusion_matrix(expected, predicted))
'''