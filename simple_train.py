import os
import numpy as np
from numpy import array
import cv2 
import time
import pickle
from pathlib import Path
d = 8
k = 3
confusion_dir = 'confusion/'
confusion_mat_dir = 'confusion_matrice/'
storage_dir = 'stockage/'
path = 'dataset3'

dirs = os.listdir(path)

print('nombre de classes : ',len(dirs))
total = 0

DATAPOINTS = []
start_time = time.time()
CLASSES = []

for idx, obj in enumerate(dirs):
	temp = []
	print('CLASSES [',idx,']=',obj, ' | images samples : ',len(os.listdir(path+'/'+obj+'/train')))
	total +=len(os.listdir(path+'/'+obj+'/train'))
	st_local = time.time()
	doss = os.listdir(path+'/'+obj+'/train')
	CLASSES.append(obj)
	for idx, fic in enumerate(doss):
		sift = cv2.xfeatures2d.SIFT_create(d)
		kp1, des1 = sift.detectAndCompute(cv2.imread(path+'/'+obj+'/train/'+fic),None)
		temp.append(des1)
		print(fic,'...........................',idx/len((os.listdir(path+'/'+obj+'/train'))))
	DATAPOINTS.append(temp)
	print("---general %s seconds ---| ---local %s seconds ---" % (time.time() - start_time, time.time() - st_local))
print('image total : ',total)


print("--- %s seconds ---" % (time.time() - start_time))

if(not Path(storage_dir+"stockage_simple.gt").is_file()):
	os.mknod(storage_dir+"stockage_simple.gt")
f = open(storage_dir+"stockage_simple.gt", "wb")
f.truncate(0)
pickler = pickle.Pickler(f)
pickler.dump([DATAPOINTS,CLASSES])
print('okay, saved')
