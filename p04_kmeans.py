from matplotlib.image import imread, imsave
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from datetime import datetime
import os

A_large = imread("data\peppers-large.tiff")
A_small = imread("data\peppers-small.tiff")

data = A_small[:, :, 2]
data = A_small.reshape(-1, 3)

# # KMEANS
cant_clusters = 16
kmeans = KMeans(n_clusters=cant_clusters)
kmeans.fit(data)
y_kmeans = pd.Series(kmeans.predict(data))

centroides = kmeans.cluster_centers_


# Obtener las dimensiones de la imagen
alto, ancho, canales = A_small.shape
# Redimensionar la matriz de píxeles para trabajar con K-Means
A_reshape = A_small.reshape(-1, canales)
# Número de clusters
n_clusters = 16
# Crear un modelo K-Means
kmeans = KMeans(n_clusters=n_clusters, n_init=30)
kmeans.fit(A_reshape)
# Etiquetas de cluster para cada píxel
etiquetas = kmeans.labels_
# Centroides de los clusters
centroides = kmeans.cluster_centers_
# Asignar el valor del centroide más cercano a cada píxel
imagen_comprimida = centroides[etiquetas].reshape(alto, ancho, canales)

plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(A_small)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.title("Imagen Comprimida")
plt.imshow(imagen_comprimida.astype(np.uint8))
plt.axis("off")
plt.show()

# ## Calculo del factor de compresión

resolucion_imagen = 512 * 512
bits_por_pixel = 3 * np.log2(256)
bits_por_pixel_compr = np.log2(16)

bits_org = resolucion_imagen * bits_por_pixel
bits_compr = resolucion_imagen * bits_por_pixel_compr

factor = bits_org / bits_compr  # = 6

print(factor)