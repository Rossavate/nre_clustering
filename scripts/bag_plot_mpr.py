import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

plt.clf()

pattern_num = [5, 15, 30, 50, 70, 90, 100]
color = ['cornflowerblue', 'blueviolet', 'green', 'red', 'teal', 'black', 'turquoise']
markers = ['s', 'o', '*', 'x', '^', 'd', '<']

for i in range(len(pattern_num)):
    y_true = np.load('../runs/bag/model-{}/allans_bag.npy'.format(pattern_num[i]))
    y_scores = np.load('../runs/bag/model-{}/allprob_bag.npy'.format(pattern_num[i]))

    precision, recall, threshold = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)

    results = np.dstack((precision, recall))
    results.shape = (len(precision),2)

    new_results = []

    for j in range(len(results)):
        if results[j,0] >= 0.3 or results[j,1] <= 0.4:
            new_results.append(results[j])

    new_results = np.array(new_results)
    new_precision = new_results[:,0]
    new_recall = new_results[:,1]

    name = 'CNN+PFA-{}'.format(pattern_num[i])
    print('average_precision of pattern %02d : %f' %(pattern_num[i], average_precision))
    plt.plot(new_recall, new_precision, color = color[i], marker=markers[i], markevery=120, lw=1,label=name)
    #plt.plot(new_recall, new_precision, color = color[i], lw=1,label=name)


plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.3, 1.0])
plt.xlim([0.0, 0.4])
#plt.title('Precision-Recall Area={}'.format(sys.argv[1]))
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig('../images/figure7.eps', bbox_inches='tight')

print('figure7.pdf over!')