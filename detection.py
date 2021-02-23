import cv2
import os
import numpy as np
from numpy import array
import pickle
import sys
from collections import Counter

fichier = str(sys.argv[1])
d = 8
k = 1
nearest = 2
confusion_dir = 'confusion/'
confusion_mat_dir = 'confusion_matrice/'
storage_dir = 'stockage/'
path = 'dataset3'
file = open(storage_dir+"stockage_"+path+"_d_"+str(d)+".gt", 'rb')
data = pickle.load(file)
varse = data[0]
classes = data[1]
sift = cv2.xfeatures2d.SIFT_create(d)
img = cv2.imread(fichier)
kp, des = sift.detectAndCompute(img,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=128)
flann = cv2.FlannBasedMatcher(index_params,search_params)
M = []
if len(des)>nearest:
			for idy,v in enumerate(varse):
				tmp = []
				for z in v:
					c1 = 0
					matches = flann.knnMatch(z,des,k=nearest)
					for i,(m,n) in enumerate(matches):
						if m.distance <0.6*n.distance:
							c1+=1
					tmp.append(c1)
					M.append([idy,c1])
			#print(M)
			M.sort(key=lambda x: x[1], reverse=True)
			#print(M)
			k_nearest = M[:k]
			print(k_nearest)
			E = []
			for b in k_nearest:
				E.append(b[0])
			dav = Counter(E)
			pred = dav.most_common(1)[0][0]
			print(' La classe predite est : ',classes[pred])

cv2.imshow("image", img)
cv2.waitKey(0)
    


