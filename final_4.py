import pandas as pd
import numpy as np
from scipy import sparse, io

from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

import matplotlib.pyplot as plt

# Gonzalez Paper (2nd)
gonz_path = './Gonzalez/gonz_df.pkl'

# # Bush Paper (3rd)
bush_path = './Bush/buschmeier.pkl'

# # Joshi Paper (4th) Paper
joshi_path = './Context_Incongruity/jc_features_df_Balanced.pkl'

''' Starts Here '''
# Input Args
base_df_pkl_path = joshi_path

# Output Args


'''Ready Features'''
# Dataset
# df = np.loadtxt('./new_label.txt')


# labels = np.array(list(df['label']))

# bf = pd.read_pickle(base_df_pkl_path)
# print(bf)

w1 = pd.read_pickle('./wembedding/wembed_1.pkl')
w1 = w1.iloc[:, 0:4]
w3 = pd.read_pickle('./wembedding/wembed_3.pkl')
w3 = w3.iloc[:, 0:4]
w5 = pd.read_pickle('./wembedding/wembed_5.pkl')
w5 = w5.iloc[:, 0:4]


# Sparse Matrices
# # Sparse Load
# bf = io.mmread('./joshi/jc_quotes.mtx')
# # labels = np.loadtxt('./new_label_balanced.txt', dtype=np.int32)
# labels = pd.read_csv('./dataset', sep='\t', names=['text', 'label'])
# print(labels)
# labels = labels['label']
# labels = np.array(list(labels))
# print(labels)
# stats_path = "./J_BASE_QUOTES"

# # # Append WordEmbedding Feature
# bf_1 = sparse.hstack((bf, w1))
# bf_3 = sparse.hstack((bf, w3))
# bf_5 = sparse.hstack((bf, w5))

# print(bf.shape)
# print(labels.sum())

# Training
# Initialize model
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
    print(metric)
    return scores


# scores = []

# temp = classify(bf, labels, svmmodel)
# scores.append(temp)

# temp = classify(bf_1, labels, svmmodel)
# scores.append(temp)

# temp = classify(bf_3, labels, svmmodel)
# scores.append(temp)

# temp = classify(bf_5, labels, svmmodel)
# scores.append(temp)


'''NEW SCRIPT'''

scores = []
stats_path = "./NEW_STATS_1/J_BASE_UNBAL"
train = io.mmread('./joshi/jc_unbal_train.mtx')
test = io.mmread('./joshi/jc_unbal_test.mtx')
train_labels = np.loadtxt('./data/unbalanced_train_labels.txt', dtype=np.int32)
test_labels = np.loadtxt('./data/unbalanced_test_labels.txt', dtype=np.int32)

print(train.shape, test.shape)

temp = classify_new(train, test, train_labels, test_labels, svmmodel)
scores.append(temp)
print(scores)

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
