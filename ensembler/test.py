
import FeaturesComputation as FC
import util
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from mlxtend.feature_selection import ColumnSelector
import random
from sklearn.preprocessing import StandardScaler
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = util.getData()


# columns = ColumnSelector(cols=util.column_selector())  # to select specific columns from the dataset
# variance_thresh = VarianceThreshold()   # Feature selector that remove all low-variance features

# totalFeatuers = len(variance_thresh.fit_transform(columns.fit_transform(X_train))[0])
# print(totalFeatuers)