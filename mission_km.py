import numpy as np
from skimage.io import imread
from sklearn.cluster import KMeans
#from sklearn.model_selection import train_test_split

# open training image
img_train = imread('img_train.jpg')

# normalize colors
img_train = np.array(img_train, dtype=np.float64) / 255

# reshape image
w, h, d = tuple(img_train.shape)
train_array = np.reshape(img_train, (w*h, d))

centers = [(0,128/255,0),(0,255/255,0),(255/255,128/255,0),(0,0,255/255),(0,0,0),(255/255,0,0)]
centers = np.array(centers, dtype=np.float64)

km = KMeans(n_clusters=6,init=centers,copy_x=False).fit(train_array)


# open testing image
img_test = imread('img_val.jpeg')

# normalize colors
img_test = np.array(img_test, dtype=np.float64) / 255

# reshape image
w, h, d = tuple(img_test.shape)
test_array = np.reshape(img_test, (w*h, d))

# predict classes
prediction = km.predict(test_array)

# calculate desmatamento percentage (class = 2)
for i in range(6):
    total = 0
    for pred in prediction:
        if pred == i:
            total+=1

    print("a porcentagem de "+str(i)+" é: %.2f" % (total*100/len(prediction))) 