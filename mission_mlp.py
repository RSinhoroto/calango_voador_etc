# import numpy for array handling
import numpy as np

# import itertools and matplot for confusion matrix plotting 
## IMPORTANT FOR TESTING THE NETWORK ##
import itertools
import matplotlib.pyplot as plt

# import sklearn for machine learning structures
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# import skimage for image processing and i/o
from skimage.io import imread
from skimage.color.colorconv import rgb2lab, rgb2hsv, rgb2ycbcr, rgb2gray

def find_in_color_list(item, color_list):
    if len(color_list) == 0:
        return -1
    else:
        for i in range(len(color_list)):
            if color_list[i][0] == item[0] and color_list[i][1] == item[1] and color_list[i][2] == item[2]:
                return i
        return -1

def plot_confusion_matrix(cm,classes,title='Confusion matrix',cmap='Blues'):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


#################### TRAINING ####################
# open training image
img_train = imread("img_train.png")

# remove unnecessary transparency column
img_train = img_train[:,:,:-1]

# convert to lab
img_train = rgb2hsv(img_train)

# normalize colors
#img_train = np.array(img_train, dtype=np.float64) / 255

# reshape image as list
w, h, d = tuple(img_train.shape)
train_array = np.reshape(img_train, (w*h, d))

# create train target for fitting the MLP classifier
train_target = np.empty((train_array.shape[0],), dtype=int)
color_list = []
for i in range(len(train_target)):
    if find_in_color_list(train_array[i], color_list) == -1:
        color_list.append(train_array[i])
    train_target[i] = find_in_color_list(train_array[i], color_list)

print(color_list)

# create MLP classifier with 1 hidden layer and one neuron for each class
mlp = MLPClassifier((6,))

# fit network using training set
print('Training with %d samples' % len(train_array))
mlp.fit(train_array, train_target)


#################### TESTING ####################
# open testing image
img_test = imread("img_test.png")

# remove unnecessary transparency column
img_test = img_test[:,:,:-1]

# convert to hsv
img_test = rgb2hsv(img_test)

# normalize colors
#img_test = np.array(img_test, dtype=np.float64) / 255

# reshape image as list
w, h, d = tuple(img_test.shape)
test_array = np.reshape(img_test, (w*h, d))

# create test target for analising the MLP classifier
test_target = np.empty((test_array.shape[0],), dtype=int)
for i in range(len(test_target)):
    test_target[i] = find_in_color_list(test_array[i], color_list)

# test network using test set
print('Testing with %d samples' % len(test_array))
test_pred = mlp.predict(test_array)

#plot confusion matrix for class score
#cm = confusion_matrix(test_target, test_pred, labels=None, sample_weight=None)
#plot_confusion_matrix(cm, classes=['Dark Green', 'Light Green', 'Black', 'Blue', 'Orange', 'Red'], 
#   title='Terrain Classification Confusion Matrix', cmap='Greens')


#################### VALIDATION ####################
# open validation image
img_val = imread("img_val.jpeg")

# convert to hsv
img_val = rgb2hsv(img_val)

# normalize colors
#img_val = np.array(img_val, dtype=np.float64) / 255

# reshape image as list
w, h, d = tuple(img_val.shape)
val_array = np.reshape(img_val, (w*h, d))

# validate network using validation set
print('Validating with %d samples' % len(val_array))
val_pred = mlp.predict(val_array)

# compare val and test
# calculate desmatamento percentage (class = 2)
for i in range(6):
    total = 0
    for pred in test_pred:
        if pred == i:
            total+=1

    print("a porcentagem de "+str(i)+" na imagem de teste é: %f" % (total*100/len(test_pred))) 

for i in range(6):
    total = 0
    for pred in val_pred:
        if pred == i:
            total+=1

    print("a porcentagem de "+str(i)+" na imagem de validação é: %f" % (total*100/len(val_pred))) 