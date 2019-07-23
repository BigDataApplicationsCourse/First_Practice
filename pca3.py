# Implementación de PCS y visualización en 3D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
from sklearn.decomposition import PCA

# Otra forma de cargar la base IRIS, recordar que Iris está en sklearn

iris = datasets.load_iris()
features = iris.data
labels = iris.target

# Obtener los 3 primer componentes principales

pca = PCA(3)
pca.fit(features)
featuresd = pca.transform(features)

plt.clf()

fig = plt.figure(1, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()


for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(
        featuresd[labels == label, 0].mean(),
        featuresd[labels == label, 1].mean() + 1.5,
        featuresd[labels == label, 2].mean(), name,
        horizontalalignment='center',
        bbox=dict(alpha=0.5, edgecolor='w', facecolor='w'))

# Reordenar los labels para agruparlos por color y visualizar los clusters

labels = np.choose(labels, [1, 2, 0]).astype(np.float)
ax.scatter(featuresd[:, 0], featuresd[:, 1], featuresd[:, 2], c=labels, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()
