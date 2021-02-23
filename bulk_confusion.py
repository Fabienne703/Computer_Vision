import seaborn as sn
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy

paths = ['dataset4', 'dataset3']
D = [2,4,8,16,32]
K = [1,2,3,5,10,15]
frate = open("RATE.gt", 'rb')
RATE = pickle.load(frate)
confusion_dir = 'confusion/'
confusion_mat_dir = 'confusion_matrice/'
storage_dir = 'stockage/'
for idi,path in enumerate(paths):
	for idx,d in enumerate(D):

		file = open(storage_dir+"stockage"+path+"_d_"+str(d)+".gt", 'rb')
		data = pickle.load(file)
		varse = data[0]
		classes = data[1]
		for idy,k in enumerate(K):
			f = open(confusion_mat_dir+"confusion_"+path+"_k"+str(k)+"_d_"+str(d)+".gt", 'rb')
			da = pickle.load(f)

			df_cm = pd.DataFrame(da, index = classes,
			                  columns = classes)
			plt.figure(figsize = (10,7))
			sn.heatmap(df_cm, xticklabels=True, yticklabels=True)
			matplotlib.rc('xtick', labelsize=10)
			matplotlib.rc('ytick', labelsize=10)
			plt.xticks(fontsize=8,rotation=90)
			plt.yticks(fontsize=8,rotation=0)
			s = 0
			t = 0
			for x in range(0,len(da)):
				for y in range(0,len(da)):
					s+=da[x][x]
					t+=numpy.sum(da[x])
			print(path,"[ d = ",d," ; k = ",k,"]",round((s/t)*100,2),'%')
			plt.suptitle("Matrice de confusion d = "+str(d)+" k ="+str(k)+" taux : "+str(round((s/t)*100,2))+'%', size=16)
			plt.savefig(confusion_mat_dir+"img_confusion_"+path+"_k"+str(k)+"_d_"+str(d)+".png")

