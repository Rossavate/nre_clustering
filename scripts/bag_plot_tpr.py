import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

plt.clf()
filename = ['Mintz','MultiR','MIML','CNN+ATT','PCNN+ATT', 'APCNN+soft-label', 'CNN+RL', 'CNN+PFA-type', 'CNN+PFA']
color = ['cornflowerblue', 'turquoise', 'darkorange', 'red', 'teal', 'black', 'darkblue', 'green', 'blueviolet']
markers = ['<', 's', 'o', '*', 'x', '^', 'd', 'v', 'p']

for i in range(len(filename)):
	precision = np.load('../data/pr_data/'+filename[i]+'_precision.npy')
	recall = np.load('../data/pr_data/'+filename[i]+'_recall.npy')
	if filename[i] == 'CNN+RL':
		precision = precision[0]
		recall = recall[0]
		plt.plot(recall, precision, color=color[i], marker=markers[i], markevery=10, lw=1.2, label=filename[i])
	else:
		plt.plot(recall, precision, color=color[i], marker=markers[i], markevery=120, lw=1, label = filename[i])


plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.3, 1.0])
plt.xlim([0.0, 0.4])
#plt.title('Precision-Recall Area={}'.format(sys.argv[1]))
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig('../images/figure4.eps',bbox_inches='tight')

print('figure.pdf over!')