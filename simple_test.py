import cv2
import os
import numpy as np
from numpy import array
import pickle
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

d = 8
k = 3
confusion_dir = 'confusion/'
confusion_mat_dir = 'confusion_matrice/'
storage_dir = 'stockage/'
path = 'dataset3'
file = open(storage_dir+"storage_simple.gt", 'rb')
data = pickle.load(file)
varse = data[0]
classes = data[1]

sift = cv2.xfeatures2d.SIFT_create(d)





path = 'dataset3'
dirs = os.listdir(path)

print('nombre de classes : ',len(dirs))
correct = 0
total = 0
nearest = 2
#start_time = time.time()
CONFUSION = [] 
for idx, obj in enumerate(dirs):
	#total +=len(os.listdir(path+'/'+obj+'/test'))
	doss = os.listdir(path+'/'+obj+'/test')
	local_conf = np.zeros(len(classes))
	for idz, fic in enumerate(doss):
		img = cv2.imread(path+'/'+obj+'/test/'+fic)
		kp, des = sift.detectAndCompute(img,None)
		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=128)
		flann = cv2.FlannBasedMatcher(index_params,search_params)

		M = []
		if len(des)>nearest:
			total +=1
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
			local_conf[pred] +=1
			if classes[pred] == obj:
				correct +=1
				#print('Originale ',obj,' La classe predite est : ',pred)
			print('Originale ',obj,' La classe predite est : ',classes[pred])
			print("Dectection de ",fic,' actual correction rate ',round((correct/total)*100,2),'%')
	CONFUSION.append(local_conf)
	#break
print('Overall result',round((correct/total)*100,2),'%')
if(not Path(confusion_dir+"confusion_simple.gt").is_file()):
	os.mknod(confusion_dir+"confusion_simple.gt")
f = open(confusion_dir+"confusion_simple.gt", "wb")
f.truncate(0)
pickler = pickle.Pickler(f)
pickler.dump(CONFUSION)

