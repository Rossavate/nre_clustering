import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

plt.clf()
filename = ['CNN+ATT','PCNN+ATT', 'APCNN+soft-label', 'PEACNN-type', 'PEACNN']
color = ['red', 'teal', 'black', 'green', 'blueviolet' ]
markers = ['o', '*', 'x', '^', 'd']

for i in range(len(filename)):
	precision = np.load('../data/pr_data/'+filename[i]+'_precision.npy')
	recall  = np.load('../data/pr_data/'+filename[i]+'_recall.npy')
	plt.plot(recall, precision, color = color[i], marker=markers[i], markevery=120, lw=1, label = filename[i])


plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.3, 1.0])
plt.xlim([0.0, 0.4])
#plt.title('Precision-Recall Area={}'.format(sys.argv[1]))
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig('../images/figure5.pdf')

print('figure5.pdf over!')