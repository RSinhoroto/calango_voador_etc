import cv2
import numpy as np
from matplotlib import pyplot as plt 

def gera_rotulo(img, new_label):
    
    img_lbl = img.copy()
    h, w = img.shape
    equival = []
    
    for i in range (1, h):
        for j in range(1, w):
            if img[i,j] == 0:
                if img[i-1,j] == 1 and img[i,j-1] == 1:
                    img_lbl[i,j] = new_label
                    new_label += 1
                elif img[i-1,j] == 1 and img[i,j-1] == 0:
                    img_lbl[i,j] = img_lbl[i,j-1]
                elif img[i-1,j] == 0 and img[i,j-1] == 1:
                    img_lbl[i,j] = img_lbl[i-1,j]
                else:
                    if img_lbl[i,j-1] == img_lbl[i-1,j]:
                        img_lbl[i,j] = img_lbl[i-1,j]
                    else:
                        maior = max(img_lbl[i,j-1], img_lbl[i-1,j])
                        menor = min(img_lbl[i,j-1], img_lbl[i-1,j])
                        if (menor, maior) not in equival:
                            equival.append((menor, maior))
                        img_lbl[i,j] = menor
    
    # resolve label equivalence
    equival.reverse()
    for p in equival:
        for i in range(h):
            for j in range(w):
                if img_lbl[i,j] == p[1]:
                    img_lbl[i,j] = p[0]
    del(equival)

    # prepare number of new label for next iteration
    new_label = img_lbl.max() + 1 

    # return label matrix and new label
    return img_lbl, new_label


def extend_img(img):
    h,w = np.shape(img)
    h1 = h+1
    w1 = w+1
    img_new = np.ones((h1, w1), dtype=int)
    img_new[1:h1, 1:w1] = img[:, :]
    return img_new


# compute total number of elements
def count_elements(img_lbl):
    vals = []       # array of labels

    for l in img_lbl:
        for p in l: 
            if p not in vals and p > 1:     # count number of different labels
                vals.append(p)

    return vals
