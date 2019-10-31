import pandas as pd
import numpy as np
from scipy import sparse, io

from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

import matplotlib.pyplot as plt

'''NEW SCRIPT'''
dataset = "unbalanced"
scores = []
stats_path = "./NEW_STATS_1/ALL_BALANCED"
lieb_train = io.mmread('./lieb/lieb_{}_train.mtx'.format(dataset))
lieb_test = io.mmread('./lieb/lieb_{}_test.mtx'.format(dataset))
gonz_train = io.mmread('./gonzalez/gonz_{}_train.mtx'.format(dataset))
gonz_train = gonz_train.tocsc()
gonz_test = io.mmread('./gonzalez/gonz_{}_test.mtx'.format(dataset))
gonz_test = gonz_test.tocsc()
bush_train = io.mmread('./bush/bush_{}_train.mtx'.format(dataset))
bush_train = bush_train.tocsc()
bush_test = io.mmread('./bush/bush_{}_test.mtx'.format(dataset))
bush_test = bush_test.tocsc()
josh_train = io.mmread('./joshi/jc_{}_train.mtx'.format(dataset))
josh_train = josh_train.tocsc()
josh_test = io.mmread('./joshi/jc_{}_test.mtx'.format(dataset))
josh_test = josh_test.tocsc()

train_labels = np.loadtxt(
    './ddata/{}_train_labels.txt'.format(dataset), dtype=np.int32)
test_labels = np.loadtxt(
    './ddata/{}_test_labels.txt'.format(dataset), dtype=np.int32)

print(lieb_test.shape, bush_test.shape, gonz_test.shape, josh_test.shape)

names = []
names.append('Excalamation')
names.append('Question_Mark')
names.append('dotdotdot')
names.append('Interjection')
names.append('Abs_Pos_Score')
names.append('Abs_Neg_Score')
names.append('Polarity')
names.append('Largest_Sequence')
names.append('Flips')
jc = ['j_'+i for i in names[-5:]]
bush = ["b_quotes", "b_hyp", "b_pnpunct", "b_pnellip", "b_exclamation", "b_question", "b_dotdotdot",
        "b_interj"]

gonz = ["g_exclamation", "g_question",
        "g_dotdotdot", "g_interj", "g_LP", "g_PP", "g_PC"]

# RFE Top 11 Features
rfe_names = [gonz[-7]] + [gonz[-5]] + gonz[-2:] + \
    bush[-8:-4] + [bush[-1]] + jc[-3:-1]
all_rfe_train = sparse.hstack((lieb_train, gonz_train[:, -7], gonz_train[:, -5],
                               gonz_train[:, -2:], bush_train[:, -8:-4], bush_train[:, -1], josh_train[:, -3:-1]))
all_rfe_test = sparse.hstack((lieb_test, gonz_test[:, -7], gonz_test[:, -5],
                              gonz_test[:, -2:], bush_test[:, -8:-4], bush_test[:, -1], josh_test[:, -3:-1]))

# Feature Importance Top 11 Features
fimp_names = [gonz[-7]] + [gonz[-5]] + gonz[-3:-1] + \
    [bush[-7]] + [bush[-5]] + [bush[-1]] + jc[-2:] + jc[-5:-3]
all_fimp_train = sparse.hstack((lieb_train, gonz_train[:, -7] + gonz_train[:, -5], gonz_train[:, -3:-1],
                                bush_train[:, -7], bush_train[:, -5], bush_train[:, -1], josh_train[:, -64:-9], josh_train[:, -2:], josh_train[:, -5:-3]))
all_fimp_test = sparse.hstack((lieb_test, gonz_test[:, -7] + gonz_test[:, -5], gonz_test[:, -3:-1],
                               bush_test[:, -7], bush_test[:, -5], bush_test[:, -1], josh_test[:, -64:-9], josh_test[:, -2:], josh_test[:, -5:-3]))

# Chi2 Top 11 Features
chi2_names = [gonz[-7]] + [gonz[-5]] + gonz[-3:] + \
    bush[-8:-4] + [bush[-1]] + [jc[-2]]

all_chi2_train = sparse.hstack((lieb_train, gonz_train[:, -7], gonz_train[:, -5],
                                gonz_train[:, -3:], bush_train[:, -8:-4], bush_train[:, -1], josh_train[:, -2]))
all_chi2_test = sparse.hstack((lieb_test, gonz_test[:, -7], gonz_test[:, -5],
                               gonz_test[:, -3:], bush_test[:, -8:-4], bush_test[:, -1], josh_test[:, -2]))


svmmodel = SGDClassifier(
    early_stopping=True, max_iter=100, n_jobs=-1, verbose=1)


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
    with open('results_{}.txt'.format(dataset), 'w') as f:
        for yp, yt in zip(y_pred, y_test):
            f.write("{}\t{}\n".format(yp, yt))
    metric = precision_recall_fscore_support(y_test, y_pred)
    scores.append(metric)
    print(metric)
    return scores


# Training
# print(rfe_names)
# temp = classify_new(all_rfe_train, all_rfe_test,
#                     train_labels, test_labels, svmmodel)
# scores.append(temp)

# print(fimp_names)
# temp = classify_new(all_fimp_train, all_fimp_test,
#                     train_labels, test_labels, svmmodel)
# scores.append(temp)

print(chi2_names)
print(all_chi2_test.shape)
temp = classify_new(all_chi2_train, all_chi2_test,
                    train_labels, test_labels, svmmodel)
scores.append(temp)

# Storing Results
for i in range(0, 3):
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
