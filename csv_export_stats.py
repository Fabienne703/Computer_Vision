import seaborn as sn
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import csv
from pathlib import Path
import os

paths = ['dataset4', 'dataset3']
confusion_dir = 'confusion/'
confusion_mat_dir = 'confusion_matrice/'
storage_dir = 'stockage/'
stat_dir = 'statistique/'

D = [2,4,8,16,32]
K = [1,2,3,5,10,15]
LAB  = []
RATE = []
for idi,path in enumerate(paths):
	labels = []
	rates = []
	for idx,d in enumerate(D):
		file = open(storage_dir+"stockage_"+path+"_d_"+str(d)+".gt", 'rb')
		data = pickle.load(file)
		for idy,k in enumerate(K):
			f = open(confusion_dir+"confusion_"+path+"_k"+str(k)+"_d_"+str(d)+".gt", 'rb')
			da = pickle.load(f)
			s = 0
			t = 0
			for x in range(0,len(da)):
				for y in range(0,len(da)):
					s+=da[x][x]
					t+=numpy.sum(da[x])
			print(path,"[ d = ",d," ; k = ",k,"]",round((s/t)*100,2),'%')
			labels.append(["d = "+str(d)+" ; k = "+str(k)])
			rates.append(round((s/t)*100,2))
	RATE.append(rates)
	LAB.append(labels)
print (RATE[0])
for i,e in enumerate(RATE):
	list1 = RATE[i]
	list2 = LAB[i]
	fig = plt.figure(figsize=(10,7))
	ax = fig.add_subplot(111)
	ax.plot(list1)
	ax.set_yticklabels(list2)

	fig.savefig(path[i]+"_graph.png")
	if(not Path(stat_dir+path[i]+"_statistique.csv").is_file()):
		os.mknod(stat_dir+path[i]+"_statistique.csv")
	with open(stat_dir+path[i]+"_statistique.csv", 'w') as myfile:
	    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
	    wr.writerow(['d & k','correction rate (%)'])
	    for ids,f in enumerate(LAB[i]):
	    	wr.writerow([LAB[i][ids],RATE[i][ids]])

