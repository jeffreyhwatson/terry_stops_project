import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline 
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import make_scorer, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_sm_pipeline

import pandas as pd
import numpy as np

from src import classes as c

import matplotlib.pyplot as plt
import seaborn as sns

def pre_score(y_true, y_pred):
    "Precision scoring function for use in make_scorer."
    
    precision = precision_score(y_true, y_pred, zero_division=0)

    return precision

# creating scorer object for pipelines
precision = make_scorer(pre_score)

def f_score(y_true, y_pred):
    "F1 scoring function for use in make_scorer."
    
    f1 = f1_score(y_true, y_pred)

    return f1

# creating scorer object for pipelines
f1 = make_scorer(f_score)

def confusion(model, X, y):
    "Returns a confusion matrix."
    
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_confusion_matrix(model, X, y,
                          cmap=plt.cm.Blues, 
                          display_labels=['Minor Outcome', 'Major Outcome'], ax=ax)
    plt.title('Confusion Matrix')
    plt.grid(False)
#     plt.savefig('LR_Final_CM',  bbox_inches ="tight",\
#                 pad_inches = .25, transparent = False)
    plt.show()

def framer(df, col, li):
    "Returns a data frame with selected columns."
    
    _list = [x for x in li if x not in col]
    column_list = df.columns
    cols = [x for x in column_list if x not in _list]
    return df[cols]

def Xy(df):
    """Returns a data frame and target series."""
    
    X = df.drop('Target', axis=1)
    y = df['Target']
    return X, y

def splitter(X, y):
    """Returns a train/test split."""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=2021,
                                                        stratify=y
                                                   )
    return  X_train, X_test, y_train, y_test

def subsplit_test(X_train, y_train, model):
    """Returns train/test scores & a confusion matrix on subsplit test data."""
    
    modeling = c.Harness(f1)
    Xs_train, Xs_test, ys_train, ys_test = splitter(X_train, y_train)
    model.fit(Xs_train, ys_train)
    train_score = f1_score(ys_train, model.predict(Xs_train))
    test_score = f1_score(ys_test, model.predict(Xs_test))
    confusion(model, Xs_train, ys_train)
    confusion(model, Xs_test, ys_test)
    recall_test = recall_score(ys_test, model.predict(Xs_test))
    precision_test = precision_score(ys_test, model.predict(Xs_test))
    report = pd.DataFrame([[train_score, test_score, recall_test, precision_test]],\
                          columns=['Train F1', 'Test F1', 'Test Recall', 'Test Precision'])
    return report
    
def feature_plot(transformer, gridsearch, X):
    """Returns feature importances of the best estimator of a gridsearch."""
    
    transformer.transform(X)
    features = list(gridsearch.best_estimator_[0].transformers_[0][1].get_feature_names())+list(X.select_dtypes('number').columns)
    importances = gridsearch.best_estimator_[1].feature_importances_
    sorted_importances = sorted(list(zip(features, importances)),key=lambda x: x[1], reverse=True)[:25]
    x = [val[0] for val in sorted_importances]
    y = [val[1] for val in sorted_importances]
    plt.figure(figsize=(20,6))
    plt.bar(x, y, color=['skyblue'])
    plt.xticks(rotation=90)
#     plt.savefig('Feature_Imp',  bbox_inches ="tight",\
#                 pad_inches = .25, transparent = False)
    plt.show()
