import numpy as np
import sys
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

y_true = np.load('../data/pr_data/allans_bag.npy')
y_scores = np.load('../data/pr_data/allprob_bag.npy')

precision, recall, threshold = precision_recall_curve(y_true, y_scores)
average_precision = average_precision_score(y_true, y_scores)

results = np.dstack((precision, recall))
results.shape = (len(precision),2)

new_results = []

for i in range(len(results)):
	if results[i,0] >= 0.3 or results[i,1] <= 0.4:
		new_results.append(results[i])

new_results = np.array(new_results)
new_precision = new_results[:,0]
new_recall = new_results[:,1]

if sys.argv[1] == 'type':
	name = 'CNN+PFA'
else:
	name = 'CNN+PFA-type'


np.save('../data/pr_data/{}_precision.npy'.format(name), new_precision)
np.save('../data/pr_data/{}_recall.npy'.format(name), new_recall)

print('average_precision_score: {0:0.5f}'.format(average_precision))