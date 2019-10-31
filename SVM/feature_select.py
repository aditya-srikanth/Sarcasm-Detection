import pandas as pd
import numpy as np
from scipy import sparse, io

from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, RFE
from sklearn.linear_model import LogisticRegression
import pickle

import matplotlib.pyplot as plt

'''NEW SCRIPT'''

scores = []
stats_path = "./NEW_STATS_1/ALL_BALANCED"
lieb_train = io.mmread('./lieb/lieb_balanced_train.mtx')
lieb_train = lieb_train.tocsc()
gonz_train = io.mmread('./gonzalez/gonz_balanced_train.mtx')
gonz_train = gonz_train.tocsc()
bush_train = io.mmread('./bush/bush_balanced_train.mtx')
bush_train = bush_train.tocsc()
josh_train = io.mmread('./joshi/jc_balanced_train.mtx')
josh_train = josh_train.tocsc()

train_labels = np.loadtxt('./ddata/balanced_train_labels.txt', dtype=np.int32)

all_feat_train = sparse.hstack(
    (lieb_train, gonz_train[:, -7:], bush_train[:, -8:-4], josh_train[:, -64:-9],  josh_train[:, -5:]))

model = LogisticRegression()
rfe = RFE(model)
print('Starting Feature Selection')
rfe = rfe.fit(all_feat_train, train_labels)
print(rfe.ranking_)
print('Finished Feature Selection')
with open('rfe_ranking', 'ab') as f:
    pickle.dump(rfe, f)

b_feature = ["b_quotes", "b_hyp", "b_pnpunct", "b_pnellip", "b_exclamation", "b_question", "b_dotdotdot",
             "b_interj"]
j_feature = ["j_exclamation", "j_question", "j_dotdotdot", "j_interj", "j_abs_pos_score", "j_abs_neg_score",
             "j_polarity", "j_largest_seq", "j_flips"]
g_feature = ["g_exclamation", "g_question",
             "g_dotdotdot", "g_interj", "g_LP", "g_PP", "g_PC"]

all_features = b_feature + j_feature + g_feature
