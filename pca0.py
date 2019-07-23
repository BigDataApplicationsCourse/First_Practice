# Implementación de PCA utilziando la librería Numpy

from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

# Cargar la base IRIS

iris = datasets.load_iris()
features = iris.data
labels = iris.target

# definimos la media (T=transpuesta de la matriz)

M = mean(features.T, axis=1)
print('mean ', M)

# centramos las columnas restando la media en cada columna

C = features - M
#print('center ', C)

# calculamos la matriz de covarianza de la matriz centrada
V = cov(C.T)
print('covariance ', V)

# eigen decomposition de la matriz de covarianza
values, vectors = eig(V)
print('vectores ', vectors)
print('valores ', values)

# proyección de los datos
P = vectors.T.dot(C.T)
print('proyeccion ', P.T)


# Cálculo de los componentes principales (PCA), utilizando la librería PCA de sklearn

from sklearn.decomposition import PCA

# creamos una instancia de PCA, con 4 PCA... puede ser 2 PCA
pca = PCA(4)

# entrenamos con los datos = features de iris
pca.fit(features)

# accesamos a los valores y vectores eigen

print('pca vectores ', pca.components_)
print('valores ', pca.explained_variance_)

# proyectamos los datos en la nueva dimensiondf = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
B = pca.transform(features)
print('trasnform ', B)

# Comparar los resultados.... deben ser muy similares.
