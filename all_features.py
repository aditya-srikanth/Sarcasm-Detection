import pandas as pd
import numpy as np
from scipy import io, sparse
import pickle

from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, RFE, f_regression
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
# Lieb Paper (1st)
lieb_path = './Lieb/lieb.pkl'

# Gonzalez Paper (2nd)
gonz_path = './Gonzalez/gonz_df.pkl'

# # Bush Paper (3rd)
bush_path = './Bush/buschmeier.pkl'

# # Joshi Paper (4th) Paper
joshi_path = './Context_Incongruity/jc_features_df.pkl'

''' Starts Here '''
# Input Args
# base_df_pkl_path = gonz_path

# Output Args
stats_path = "./stats/FULL_BASE"


'''Ready Features'''
# Dataset
df = pd.read_csv('final_data.tsv', sep='\t')
labels = np.array(list(df['label']))

# Paper Features
# lieb = io.mmread('./Lieb/lieb.mtx')

gonz = pd.read_pickle(gonz_path)
print(gonz.shape)
# gonz = gonz.iloc[:, -10:]
gonz = sparse.csr_matrix(gonz.values)

bush = pd.read_pickle(bush_path)
# bush = bush.iloc[:, -10:]
bush = sparse.csr_matrix(bush.values)

josh = pd.read_pickle(joshi_path)
# josh = josh.iloc[:, -10:]
josh = sparse.csr_matrix(josh.values)

# print(lieb.shape, gonz.shape, bush.shape, josh.shape)

# Append all Features
full = sparse.hstack((gonz, bush, josh))
print(full.shape)

# Training
# Initialize model
svmmodel = svm.SVC(gamma='scale', class_weight='balanced',
                   C=20.0, cache_size=4000)


def classify(data, labels, model):
    # 10 is pseudo random number
    kfold = KFold(5, True, 101)
    scores = []
    i = 0
    for train, test in kfold.split(data):
        print(i)
#         dd = SparseRowIndexer(data)
        data = data.tocsr()
        X_train, X_test, y_train, y_test = data[train,
                                                :], data[test, :], labels[train], labels[test]
        model.fit(X_train, y_train.ravel())
        y_pred = model.predict(X_test)
        metric = precision_recall_fscore_support(y_test, y_pred)
        scores.append(metric)
        print("Done Interation: %d" % (i))
        print(scores)
        i += 1
    return scores


# TRAINING

scores = []
print(full.shape)
temp = classify(full, labels, svmmodel)
scores.append(temp)

for i in range(0, 1):
    plt.title('Performance')
    plt.plot([x for x in range(len(scores[i]))], [score[0][1]
                                                  for score in scores[i]])
    plt.plot([x for x in range(len(scores[i]))], [score[1][1]
                                                  for score in scores[i]])
    plt.plot([x for x in range(len(scores[i]))], [score[2][1]
                                                  for score in scores[i]])
    plt.legend(['Precison', 'Recall', 'F1-Score'])
    plt.savefig(stats_path+str(i))
    # plt.show()

    confidence = []
    p_scores = [score[0][1] for score in scores[i]]
    r_scores = [score[1][1] for score in scores[i]]
    f_scores = [score[2][1] for score in scores[i]]
    print(np.array(p_scores).mean())
    print(np.array(r_scores).mean())
    print(np.array(f_scores).mean())
    # confidence intervals
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(f_scores, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(f_scores, p))
    print('%.1f confidence interval: %.1f and %.1f' %
          (alpha*100, lower*100, upper*100))
    with open(stats_path+'_stats_'+str(i)+'.txt', 'w') as f:
        f.write('F-Score Mean: %f \n' % (np.array(f_scores).mean()))
        f.write('P-Score Mean: %f \n' % (np.array(p_scores).mean()))
        f.write('R-Score Mean: %f \n' % (np.array(r_scores).mean()))
        f.write('%.1f confidence interval: %.1f and %.1f \n' %
                (alpha*100, lower*100, upper*100))
        f.write('Precision: ')
        f.write(' '.join(str(x) for x in p_scores))
        f.write('\n')
        f.write('Recall: ')
        f.write(' '.join(str(x) for x in r_scores))
        f.write('\n')
        f.write('F-Score: ')
        f.write(' '.join(str(x) for x in f_scores))
        f.write('\n')
