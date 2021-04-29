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

import matplotlib.pyplot as plt
import seaborn as sns

def pre_score(y_true, y_pred):
    
    precision = precision_score(y_true, y_pred, zero_division=0)

    return precision

precision = make_scorer(pre_score)

def f_score(y_true, y_pred):
    "f1 scoring funcion for use in make_scorer."
    f1 = f1_score(y_true, y_pred)

    return f1

# creating scorer object for pipelines
f1 = make_scorer(f_score)

def confusion(model, X, y):
    "Confusion matrix plotting aid."
    model.fit(X, y)
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_confusion_matrix(model, X, y,
                          normalize='all', cmap=plt.cm.Blues,
                          display_labels=['No Arrest', 'Arrest'], ax=ax)
    plt.title('Confusion Matrix')
    plt.grid(False)
    plt.show()

def framer(df, col, li):
    _list = [x for x in li if x not in col]
    column_list = df.columns
    cols = [x for x in column_list if x not in _list]
    return df[cols]

def Xy(df):
    X = df.drop('Target', axis=1)
    y = df['Target']
    return X, y

def splitter(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=2021,
                                                    stratify=y
                                                   )
    return  X_train, X_test, y_train, y_test