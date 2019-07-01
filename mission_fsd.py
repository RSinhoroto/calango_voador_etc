import cv2
import numpy as np
import imgProcLib
from skimage.io import imread, imshow, show
from skimage.color.colorconv import rgb2hsv, rgb2lab, rgb2ycbcr
from matplotlib import pyplot as plt 

# open image for detection
img = imread('img_train.png')

# remove unnecessary transparency column
img = img[:,:,:-1]

# convert to HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# red mask
low_red = np.array([0, 200, 200])
high_red = np.array([10, 255, 255])
red_mask = cv2.inRange(hsv_img, low_red, high_red)
red_img = cv2.bitwise_and(img, img, mask=red_mask)

# map mask into binary values
h,w = np.shape(red_mask)
for i in range (h):
	for j in range(w):
		red_mask[i,j] /= 255

red_mask = 1-red_mask
red_mask_ext = imgProcLib.extend_img(red_mask)
red_mask_lbl, label = imgProcLib.gera_rotulo(red_mask_ext, 2)

red_mask_holes = imgProcLib.extend_img(red_mask_ext)
red_mask_holes = 1-red_mask_holes
holes_lbl, label = imgProcLib.gera_rotulo(red_mask_holes, label)


# compute total number of elements
vals = imgProcLib.count_elements(red_mask_lbl)
holes = imgProcLib.count_elements(holes_lbl)
vals.insert(0,1) # add background to list of values


# mark centers and calculate areas
new_img = img.copy()
for lbl in vals:
	points = np.where(red_mask_lbl == lbl)
	ylim = (np.min(points[0]), np.max(points[0]))
	xlim = (np.min(points[1]), np.max(points[1]))

	print(xlim, ylim)
	
	ISFIRE = True
	for hole in holes:
		print("element {0} looking for hole {1}".format(lbl, hole))
		found = np.where(holes_lbl[xlim[0]+1:xlim[1]+1, ylim[0]+1:ylim[1]+1]==hole)
		if np.size(found[0]) > 0:
			print('Found!\n')
			ISFIRE = False
			break

	if ISFIRE:
		center = (int((xlim[0]+xlim[1])/2), int((ylim[0]+ylim[1])/2))
		diameter = max(xlim[1]-xlim[0], ylim[1]-ylim[0])
		cv2.circle(new_img, center, int(diameter/2), (255,255,255), 5)

imshow(new_img)
show()
	


	
