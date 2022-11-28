# Proyecto Inteligencia Artificial 
**Diagnóstico del cáncer de mama mediante diferentes modelos ML**

| Tabla de contenido  
| ------------- 
| Acerca del dataset  
| Procedimiento
| Modelos 
| Procedimiento
| Conclusiones

# Acerca del dataset

El cáncer de mama es una enfermedad en la que las células de la mama crecen de forma descontrolada. Hay diferentes tipos de cáncer de mama. El tipo de cáncer de mama depende de las células de la mama que se convierten en cáncer, estos se pueden clasificar entre benigno y maligno. 

En este documentos se utilizan etiquetas y características las cuales permiten diferenciar que tipo de cáncer, así mismo se eligieron datos numéricos para el análisis de estos. 

![image](https://user-images.githubusercontent.com/119226341/204189027-3fb1fc46-338a-489f-9daf-de8666447422.png)

A continuación se presenta el procedimiento para identificación y desarrollo de los modelos: 

**Regresión Logística**

La regresión logística en dicho caso se utiliza a partir del  modo OVR, Esta clase implementa la regresión logística regularizada, solo admiten la regularización L2 con formulación primaria o sin regularización. El solucionador 'liblineal' admite la regularización de L1 y L2, con una formulación dual solo para la penalización de L2.

En cuanto a la implementación de dicho código, lo primero que se realiza es la inicialización de las librerías, luego de esto, se inicializan las variables para la evualuación de dichos métodos, es decir ACC, MCC, TPR y f1_score. 

**Accuracy (ACC):** En la clasificación multietiqueta, esta función calcula la precisión del subconjunto: el conjunto de etiquetas previsto para una muestra debe coincidir exactamente con el conjunto de etiquetas correspondiente. Que en este caso corresponde a y_predicted, es decir el diagnostico o el tipo de cáncer.


**Matthews_corrcoef (MCC):** El MCC es en esencia un valor de coeficiente de correlación entre -1 y +1. Un coeficiente de +1 representa una predicción perfecta, 0 una predicción aleatoria promedio y -1 una predicción inversa.

**F1_score:** La puntuación F1 se puede interpretar como una media armónica de precisión y recuperación, donde una puntuación F1 alcanza su mejor valor en 1 y su peor puntuación en 0.

# Importación de Librerias


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sb
%matplotlib inline
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
```
**Importación de datos**

```python
dataframe = pd.read_csv(r"proyecto.csv",sep=';')
dataframe.head(-2)
dataframe.describe()
```
Luego se puede observar el número de características y etiquetas que son seleccionadas 

```python
dataframe.shape
```
```python
print(dataframe.groupby('diagnosis').size())
```

diagnosis
B    357
M    212
dtype: int64

**Partición de datos**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,train_size=0.7, random_state=0)
print(X_train.shape)
print(X_test.shape)
scaler = StandardScaler()# Ejercicio, no use la escalización de los datos a ver que tal funciona!
scaler.fit(X_train)# el fit de los datos solo se hace con el conjunto de entrenamiento!
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

**Desarrollo para implementación de regresión logística**

```python
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
```
1. ACC_RL =  [0.9707602339181286]
2. MCC_RL =  [0.9374499319073584]
3. f1_score_RL =  0.9707602339181286

![image](https://user-images.githubusercontent.com/119226341/204192076-56430393-10d1-4bd4-8ac8-90781d9a49e6.png)


**Para la verificación del modelo se utiliza el modelo de matriz de confusión**

```python
cm = confusion_matrix(y_test, y_predicted)
#visualize
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot = True, linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)
plt.xlabel("y_predicted")
plt.ylabel("y_test")
plt.show()
```

![image](https://user-images.githubusercontent.com/119226341/204193484-9befc849-710f-4c1d-83ae-c85358d6cf47.png)


![image](https://user-images.githubusercontent.com/119226341/204191825-07a0a0e5-145b-4d77-a21d-d113c6e84639.png

**Exactitud:** 
((101+59))/(101+7+4+59)=93.56%

**Precision:** 
59/(59+7)=0.89→89.39%

**Sensibilidad:** 
59/(59+4)=0.936→93.65%

**Especifidad:**
101/(101+7)=0.935→93.52%



# Conclusiones

1. Teniendo en cuenta los resultados anteriormente presentados, se puede decir que el mejor modelo para una buena estimación en cuanto a la identificación del tipo de cáncer, es decir entre benigno y maligno, es el método de regresión logística, con un valor de 0.9707602339181286, esto debido a la información presentada en la parte teórica de este, entre más cercano este valor sea a 1 quiere decir que la estimación es mucho mejor. 
2. A partir de los valores de matriz de confusión se puede decir de igual manera cual es el mejor modelo para la representación de los datos, teniendo en cuenta parámetros tales como la exactitud, precisión sensibilidad y especificidad. 

Video
https://youtu.be/oNU31vg1mNo
