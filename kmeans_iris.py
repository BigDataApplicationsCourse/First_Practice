# el algoritmo K-means tiene los siguientes pasos:
# 1. Escoger el parámetro k (número de clusters o grupos)
# 2. Inicializar k ‘centroide’ (punto de arranque) en nuestros datos
# 3. Crear los clusters. Asignar cada punto a centroide más cercano
# 4. Mover cada centroide al centro del su cluster
# 5. Repetir el paso 3–4 hasta que los centroides convergan.

# Utilicemos el dataset Iris de pétalos de flores disponible en scikit learn, para clasificarlas utilizando clustering o clasificación multi-clase, multi-variable. Este dataset consiste de 50 muestras de cada una de las 3 especies de flores disponibles en Iris (Iris setosa, Iris virginica e Iris versicolor). Cada muestra contiene 4 características o "features": longitud y ancho de los sépalos y pétalos.

# Empezamos importando las librerías que vamos utilizar.

from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Cargamos el dataset

iris = datasets.load_iris()

# Load_iris retorna dos objetos: data y target. Vamos a llamar "X" (la entrada) a ‘data’, que contiene los datos de entrenamiento, "y" (la salida) a ‘target’, que contiene las etiquetas (o labels de clasificación). "y", contiene ‘target_names’, el significado de los labels, ‘feature_names’, la descripción de los labels ‘DESCR’, la descripción del dataset, ‘filename’.

X = iris.data[:, :2]
y = iris.target

# siempre es bueno visualizar los datos

plt.scatter(X[:,0], X[:,1], c=y, cmap='gist_rainbow')
plt.xlabel('Longitud del Sépelo', fontsize=18)
plt.ylabel('Ancho del Sépalo', fontsize=18)

plt.show()

# Ahora necesitamos definir k, el numero de clusters. Y con eso ya podemos generar una instancia y entrenar el modelo K-means.
# 3 Clusters y random_state 21 para que la generacion de centrodides aleatorio sea deterministico.
# Creamos una instancia

km = KMeans(n_clusters = 3, random_state=10)

#KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
#  n_clusters=3, n_init=10, precompute_distances='auto',
#  random_state=10, tol=0.0001, verbose=0)

# Entrenamiento con el conjunto de datos X

km.fit(X)

# Identificamos los puntos centro de los datos

centers = km.cluster_centers_

print(centers)

# A qué cluster pertenece el nuevo dato observado.

new_labels = km.labels_

# Plot el cluster identificado y compare con los resultados

fig, axes = plt.subplots(1, 2, figsize=(16,8))
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='gist_rainbow',edgecolor='k', s=150)
axes[1].scatter(X[:, 0], X[:, 1], c=new_labels, cmap='jet',edgecolor='k', s=150)
axes[0].set_xlabel('Longitud del Sépelo', fontsize=18)
axes[0].set_ylabel('Ancho del Sépalo', fontsize=18)
axes[1].set_xlabel('Longitud del Sépelo', fontsize=18)
axes[1].set_ylabel('Ancho del Sépalo', fontsize=18)
axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('Predicho', fontsize=18)

plt.show()
