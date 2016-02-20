from skimage.io import imread
from skimage import img_as_float
from sklearn.cluster import KMeans
import numpy as np

image = imread('parrots.jpg')
float_image = img_as_float(image)
p = np.array(float_image)
s = p.reshape(713 * 474, 3)
pass
kmeans = KMeans(random_state=241, init='k-means++')
kmeans.fit(s)
print(kmeans.n_clusters)
