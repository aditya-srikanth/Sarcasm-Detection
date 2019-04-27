import pandas as pd
import numpy as np
from scipy import sparse, io

from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

import matplotlib.pyplot as plt
import pickle

'''NEW SCRIPT'''

model_path = "JOSHI_MODEL.pkl"
scores = []
stats_path = "./NEW_STATS_1/J_EXPLAIN"
train = io.mmread('./joshi/jc_expl_train.mtx')
test = io.mmread('./joshi/jc_expl_test.mtx')
train_labels = np.loadtxt('./data/explain_labels.txt', dtype=np.int32)
test_labels = np.loadtxt('./data/explain_labels_test.txt', dtype=np.int32)


print(train.shape, test.shape)

svmmodel = svm.SVC(gamma='scale', class_weight='balanced',
                   C=20.0, cache_size=1000)


def classify(data, labels, model):
    # 10 is pseudo random number
    kfold = KFold(5, True, 101)
    scores = []
    i = 0
    for train, test in kfold.split(data):
        print(i)
        data = data.tocsr()
        X_train, X_test, y_train, y_test = data[train,
                                                :], data[test, :], labels[train], labels[test]
        model.fit(X_train, y_train.ravel())
        y_pred = model.predict(X_test)
        metric = precision_recall_fscore_support(y_test, y_pred)
        scores.append(metric)
        print("Done Interation: %d" % (i))
        print(metric)
        i += 1
    return scores


def classify_new(X_train, X_test, y_train, y_test, model):
    print('Started Training')
    X_train = X_train.tocsr()
    X_test = X_test.tocsr()
    scores = []
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)
    metric = precision_recall_fscore_support(y_test, y_pred)
    scores.append(metric)
    pickle.dump(model, open(model_path, 'wb'))
    print(metric)
    return scores


# Training
temp = classify_new(train, test, train_labels, test_labels, svmmodel)
scores.append(temp)

# Storing Results
for i in range(0, 1):
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
