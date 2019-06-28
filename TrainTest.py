import  sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import math

p = pd.read_csv("/home/vsriharshini/StressPro/TrainTestSet.csv")
col = [] 
for item in p.columns:
    if item!='Stress':
        col.append(item)
    
data = p[col]
target = p['Stress']
model = ['NaiveBayes','Linear SVM','Decision Tree','Logistic Regression']
accuracy = []
f1_score = []
#data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.3, random_state = 10)
################################################################

gnb = GaussianNB()
scores = cross_val_score(gnb,data,target, cv=10)
accuracy.append(scores.mean())

svc_model = LinearSVC(random_state=0)
scores = cross_val_score(svc_model,data,target, cv=10)
accuracy.append(scores.mean())

clf = tree.DecisionTreeClassifier()
scores = cross_val_score(clf,data,target, cv=10)
accuracy.append(scores.mean())

clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
scores = cross_val_score(clf,data,target, cv=10)
accuracy.append(scores.mean())


gnb = GaussianNB()
scores = cross_val_score(gnb,data,target, cv=10,scoring = 'f1')
f1_score.append(scores.mean())

svc_model = LinearSVC(random_state=0)
scores = cross_val_score(svc_model,data,target, cv=10,scoring = 'f1')
f1_score.append(scores.mean())

clf = tree.DecisionTreeClassifier()
scores = cross_val_score(clf,data,target, cv=10,scoring = 'f1')
f1_score.append(scores.mean())

clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
scores = cross_val_score(clf,data,target, cv=10,scoring = 'f1')
f1_score.append(scores.mean())



ind = np.arange(4)
plt.plot(ind,accuracy,color = 'b',label = 'Accuracy',marker='o',markerfacecolor='k',markersize=6)
plt.plot(ind,f1_score,color = 'g',label = 'F1_scores',marker='o',markerfacecolor='k',markersize=6)
plt.xticks(ind,model)
plt.xlabel('Model')
plt.ylabel('Mean of Scores')
plt.legend()
plt.show()
       