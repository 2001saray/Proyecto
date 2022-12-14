# -*- coding: utf-8 -*-
"""Proyecto_final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1f8X8CHFUDedo9V05YL5HSvehESD75QSm

# **Proyecto final Inteligencia Artificial**
***Saray Andrea Isaza Vides***
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sb
# %matplotlib inline
plt.rcParams['figure.figsize'] = (10, 5)
plt.style.use('ggplot')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric

from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# regresión Lógistica

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier

#SVM
from sklearn import svm

dataframe = pd.read_csv(r"proyecto.csv",sep=';')
dataframe.head(-2)
dataframe.describe()

# number of rows and columns in the dataset
dataframe.shape

print(dataframe.groupby('diagnosis').size())

"""**Conjunto de etiquetas y características**"""

X = dataframe[['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_worst']].values
y = dataframe['diagnosis'].values

"""**Partición de datos para validación y entrenamiento**"""

#partición del conjunto de muestras en validación y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,train_size=0.7, random_state=0)
print(X_train.shape)
print(X_test.shape)
scaler = StandardScaler()# Ejercicio, no use la escalización de los datos a ver que tal funciona!
scaler.fit(X_train)# el fit de los datos solo se hace con el conjunto de entrenamiento!
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

"""**Regresión logística**"""

#OVR
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns # data visualization library


ACC_RL=[]
MCC_RL=[]
TPR_RL=[]

# clasificador de tipo clase, multiclase de tipo OVR con un maximo de iteraciones 100
metrica = LogisticRegression(penalty='l2',max_iter=1000, C=1000,random_state=0)
metrica.fit(X_train, y_train)
y_predicted=metrica.predict(X_test)
y_score=metrica.predict_proba(X_test)


#Hallar: Accuracy
ACC_RL.append(metrica.score(X_test, y_test))
print("ACC_RL = ",ACC_RL)
#MCC 
MCC_RL.append(matthews_corrcoef(y_test,y_predicted))
print("MCC_RL = ",MCC_RL)
#f1_score
modelo=f1_score(y_test, y_predicted, average='micro')
print("f1_score_RL = ",modelo)

"""**Evaluación del modelo**"""

cm = confusion_matrix(y_test, y_predicted)

#visualize
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot = True, linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)
plt.xlabel("y_predicted")
plt.ylabel("y_test")
plt.show()

model_evento = pd.DataFrame({'Model: Regresión logística': ['ACC_RL','MCC_RL','f1_score_RL'], 
                         'Accuracy': [ACC_RL,MCC_RL,modelo]})
model_evento

plt.figure(figsize=(8,10))
plt.title("Represent Accuracy of different models")
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
plt.xticks(rotation=45)
plt.bar(model_evento['Model: Regresión logística'],model_evento['Accuracy'])
plt.show()

"""**Clasificador KNN**"""

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import f1_score

k_range = range(1, int(np.sqrt(len(y_train))))
distance='minkowski'#podemos hacer un for que recorra las distancias que queremos probar en un enfoque grid-search.

ACC_KNN=[]
MCC_KNN=[]
TPR_KNN=[]




for k in k_range:#por ahora variemos K, 
    knn = KNeighborsClassifier(n_neighbors = k,weights='distance',metric=distance, metric_params=None,algorithm='brute')
    #knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_predicted=knn.predict(X_test)
    y_score=knn.predict_proba(X_test)
    #Hallar: Accuracy
    ACC_KNN.append(knn.score(X_test, y_test))
    #MCC 
    MCC_KNN.append(matthews_corrcoef(y_test,y_predicted))
    #f1
    #f1_score(y_test, y_predicted, average='micro')
    TPR_KNN.append(recall_score(y_test,y_predicted, average='macro'))

cm = confusion_matrix(y_test, y_predicted)

#visualize
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot = True, linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)
plt.xlabel("y_predicted")
plt.ylabel("y_test")
plt.show()

"""**ACC**"""

print(ACC_KNN)
plt.figure()
plt.plot(k_range, ACC_KNN, color='orange',lw=1)
plt.xlabel('K')
plt.ylabel('Metric')
plt.title('ACC METRIC')
plt.show()

"""**MCC**"""

print(MCC_KNN)
plt.figure()
plt.plot(k_range, MCC_KNN, color='navy',lw=1)
plt.xlabel('K')
plt.ylabel('Metric')
plt.title('MCC METRIC')
plt.show()

"""**TPR**"""

print(TPR_KNN)
plt.figure()
plt.plot(k_range, TPR_KNN, color='navy',lw=1)
plt.xlabel('K')
plt.ylabel('Metric')
plt.title('TPR METRIC')
plt.show()

"""**Evaluación de modelo KNN**"""

model_evnto = pd.DataFrame({'1': ['ACC_KNN','MCC_KNN','TPR_KNN'], 
                         'Accuracy': [ACC_KNN,MCC_KNN,TPR_KNN]})
model_evnto

"""**SVM**"""

#LINEAL
kernels=['linear', 'poly', 'rbf']
#lineal
Kernel=0
msv = svm.SVC(kernel=kernels[Kernel],gamma=0.01)
#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

ACC_SVM_L=[]
MCC_SVM_L=[]
TPR_SVM_L=[]

msv.fit(X_train, y_train)

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import f1_score

y_predicted = msv.predict(X_test)
y_test_scores = msv.decision_function(X_test)
MCC_SVM_L = matthews_corrcoef(y_test, y_predicted)
print("MCC", MCC_SVM_L)
ACC_SVM_L = accuracy_score(y_test, y_predicted)
print("ACC", ACC_SVM_L)
TPR_SVM_L.append(recall_score(y_test,y_predicted, average='macro'))
print("TPR", TPR_SVM_L)
#f1_sc2ore
f1_score_SVM_L=f1_score(y_test, y_predicted, average='micro')
print("f1_score",f1_score_SVM_L)

cm = confusion_matrix(y_test, y_predicted)

#visualize
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot = True, linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)
plt.xlabel("y_predicted")
plt.ylabel("y_test")
plt.show()

"""**Evaluación de modelo SVM lineal**"""

model_evento = pd.DataFrame({'Model: SVM lineal': ['ACC_SVM_L','MCC_SVM_L','TPR_SVM_L','f1_score_SVM_L'], 
                         'Accuracy': [ACC_SVM_L,MCC_SVM_L,TPR_SVM_L,f1_score_SVM_L]})
model_evento

"""**SVM Polinomial**"""

kernels=['linear', 'poly', 'rbf']

#polinomial cuadrático
Kernel=1
msv = svm.SVC(kernel=kernels[Kernel],degree=4,coef0=1)

ACC_SVM_P=[]
MCC_SVM_P=[]
TPR_SVM_P=[]

#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

msv.fit(X_train, y_train)

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,roc_auc_score

y_predicted = msv.predict(X_test)
y_test_scores = msv.decision_function(X_test)
MCC = matthews_corrcoef(y_test, y_predicted)
print("MCC", MCC_SVM_P)
ACC = accuracy_score(y_test, y_predicted)
print("ACC", ACC_SVM_P)
TPR_SVM_P.append(recall_score(y_test,y_predicted, average='macro'))
print("TPR", TPR_SVM_P)
#f1_sc2ore
f1_score_SVM_P=f1_score(y_test, y_predicted, average='micro')
print("f1_score",f1_score_SVM_P)


cm = confusion_matrix(y_test, y_predicted)

#visualize
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot = True, linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)
plt.xlabel("y_predicted")
plt.ylabel("y_test")
plt.show()

"""**Evaluación de modelo**"""

model_evento = pd.DataFrame({'Model: SVM polinomial': ['ACC_SVM_P','MCC_SVM_P','TPR_SVM_P','f1_score_SVM_P'], 
                         'Accuracy': [ACC_SVM_P,MCC_SVM_P,TPR_SVM_P,f1_score_SVM_P]})
model_evento

"""**RBF**"""

kernels=['linear', 'poly', 'rbf']
#rbf 
Kernel=2
msv = svm.SVC(kernel=kernels[Kernel],gamma=1)#cambiar el valor de gamma
#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
msv.fit(X_train, y_train)
ACC_SVM_R=[]
MCC_SVM_R=[]
TPR_SVM_R=[]

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import f1_score

y_predicted = msv.predict(X_test)
y_test_scores = msv.decision_function(X_test)
MCC_SVM_R = matthews_corrcoef(y_test, y_predicted)
print("matthews_corrcoef", MCC_SVM_R)
ACC_SVM_R = accuracy_score(y_test, y_predicted)
print("Accuracy", ACC_SVM_R)
TPR_SVM_R.append(recall_score(y_test,y_predicted, average='macro'))
print("TPR", TPR_SVM_R)
#f1_sc2ore
f1_score_SVM_R=f1_score(y_test, y_predicted, average='micro')
print("f1_score",f1_score_SVM_R)

cm = confusion_matrix(y_test, y_predicted)

#visualize
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot = True, linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)
plt.xlabel("y_predicted")
plt.ylabel("y_test")
plt.show()

"""**Evalución del modelo SVR rbf**"""

model_evento = pd.DataFrame({'Model: SVM rbf': ['ACC_SVM_R','MCC_SVM_R','TPR_SVM_R','f1_score_SVM_R'], 
                         'Accuracy': [ACC_SVM_R,MCC_SVM_R,TPR_SVM_R,f1_score_SVM_R]})
model_evento

"""**Evaluación de general de los modelos**"""

plt.figure(figsize=(8,10))
plt.title("Represent Accuracy of different models")
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
plt.xticks(rotation=45)
plt.bar(model_evento['Model: 1'],model_evento['Accuracy'])
plt.show()