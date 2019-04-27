import pickle
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

model = pickle.load(open('JOSHI_MODEL.pkl', 'rb'))
print(model.intercept_)
