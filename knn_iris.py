# KNN (K-Nearest Neighbor) es un algoritmo supervisado de clasificación (puede también ser utilizado para regresión). El algoritmo tiene 4 pasos:
# 1. Calcula la distancia entre un nuevo dato con cada ejemplo de entrenamiento
# 2. Calcular la distnacia, para ello podemos utilizar Euclidean, Hamming o distancia de Manhattan
# 3. El modelo escoje las K filas más cercanas al nuevo dato en el dataset
# 4. Luego selecciona la clase con más puntos; esto es, la clase/label más común entre esas k filas será la clase a la que pertenece ese nuevo punto.
# Paso 1: Importar los datos y verificar los features

# importar el dataset Iris

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# crear el objeto con la base iris y sus atributos

iris = load_iris()
iris.data

# features (nombre de las columnas)

print (iris.feature_names)

# las especies están representadas con números enteros: 0=setosa; 1=versicolor; 2=virginica

print (iris.target)

# 3 tipos de labels o targets

print (iris.target_names)

# El data set esta formado por 150 ejemplos y 4 features

print (iris.data.shape)

# Paso 2: Dividir el dataset en train, test; y entrenar el modelo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Vamos a utilizar la clase KNeighborsClassifer, entonces debemos crear una instancia de esa clase, que la vamos a llamar ‘knn’; este es un objeto ‘knn’ que sabe como hacer la clasificación KNN, una vez que se le provea los datos. El parámetro ‘n_neighbors’ es el parameter/hyper parameter (k). Los otros parámetros los dejamos con los valores default

# Shape of train y test

print('Shape de X_train', X_train.shape)
print('Shape de X_test', X_test.shape)

#shape de objetos y
print('Shape de y_train', y_train.shape)
print('Shape de y_test', y_test.shape)

#importar la clse KNeihgborsClassifier from sklearn
from sklearn.neighbors import KNeighborsClassifier

#importar el modelo metrics para verificar accuracy
from sklearn import metrics

# probar desde k=1 hasta 25 y registrar la precision en testing
#‘fit’ method is used to train the model on training data (X_train,y_train) and ‘predict’ method to do the testing on testing data (X_test). Choosing the optimal value of K is critical, so we fit and test the model for different values for K (from 1 to 25) using a for loop and record the KNN’s testing accuracy in a variable (scores).

k_range = range(1,26)
scores={}
scores_list=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt

#plot las relaciones entre k y la precisión en testing
plt.plot(k_range, scores_list)
plt.xlabel('Valor de K para KNN')
plt.ylabel('Precisión en Testing')
plt.show()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
#KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1,
# n_neighbors=4, p=2, weights='uniform')

#0=setosa, 1=versicolor, 2=virginica
classes ={0:'setosa', 1:'versicolor', 2:'virginica'}

#Predicciones en datos no vistos antes
#Predecir para las 2 observaciones aleatorias siguientes:

x_new = [[3,4,5,2],
         [5,4,2,2]]

y_predict = knn.predict(x_new)

#print(classes[y_predict[0]])
print(y_predict[0])

#print(classes[y_predict[1]])
print(y_predict[1])
