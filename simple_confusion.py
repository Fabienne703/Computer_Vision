import pickle

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

d = 8
k = 3
confusion_dir = 'confusion/'
confusion_mat_dir = 'confusion_matrice/'
storage_dir = 'stockage/'
path = 'dataset3'
f = open(storage_dir+"storage_simple.gt", 'rb')

file = open(confusion_dir+"confusion_simple.gt", 'rb')
data = pickle.load(file)

da = pickle.load(f)
classes = da[1]
a = list(range(len(classes)))

df_cm = pd.DataFrame(data, index = classes,
                  columns = classes)
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, xticklabels=True, yticklabels=True)
matplotlib.rc('xtick', labelsize=3)
matplotlib.rc('ytick', labelsize=3)
plt.xticks(fontsize=5,rotation=90)
plt.yticks(fontsize=5,rotation=0)
plt.show()

