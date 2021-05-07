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
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_confusion_matrix(model, X, y,
                          cmap=plt.cm.Blues, 
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




# def feature_test(df, model, feature_list):
#     string_selector = make_column_selector(dtype_include='object')
#     number_selector = make_column_selector(dtype_include='number', dtype_exclude='object')
#     preprocessing = make_column_transformer((OneHotEncoder
#                                              (handle_unknown='ignore'),string_selector),
#                                             (StandardScaler(), number_selector))
#     modeling = c.Harness(f1)
#     for feature in feature_list:
#         feature_df = framer(df, [feature], feature_list)
#         X, y = Xy(feature_df)
#         X_train, X_test, y_train, y_test = splitter(X,y)
#         feature_pipe = make_pipeline(preprocessing, model)
#         modeling.report(feature_pipe, X_train, y_train,\
#                         f'{model} {feature} Model', f'{feature} added')
#     return modeling.history


# def feature_test_sm(df, model, feature_list):
#     string_selector = make_column_selector(dtype_include='object')
#     number_selector = make_column_selector(dtype_include='number', dtype_exclude='object')
#     preprocessing = make_column_transformer((OneHotEncoder
#                                              (handle_unknown='ignore'),string_selector),
#                                             (StandardScaler(), number_selector))
#     sm = SMOTE(random_state=2021)
#     modeling = c.Harness(f1)
    
#     for feature in feature_list:
#         feature_df = framer(df, [feature], feature_list)
#         X, y = Xy(feature_df)
#         X_train, X_test, y_train, y_test = splitter(X,y)
#         feature_pipe = make_sm_pipeline(preprocessing, sm, model)
#         modeling.report(feature_pipe, X_train, y_train,\
#                         f'{model} {feature} Model', f'{feature} added')
#     return modeling.history        


def subsplit_test(X_train, y_train, model):
    modeling = c.Harness(f1)
    Xs_train, Xs_test, ys_train, ys_test = splitter(X_train, y_train)
    model.fit(Xs_train, ys_train)
    train_score = f1_score(ys_train, model.predict(Xs_train))
    test_score = f1_score(ys_test, model.predict(Xs_test))
    confusion(model, Xs_train, ys_train)
    confusion(model, Xs_test, ys_test)
    report = pd.DataFrame([[train_score, test_score]], columns=['Train F1', 'Test F1'])
    return report
    
# def plot_feature_importances(model, data):
#     n_features = data.shape[1]
#     plt.figure(figsize=(8, 10))
#     plt.barh(range(n_features), model.feature_importances_,
#              align='center', color=['skyblue', 'darkblue'])
#     plt.yticks(np.arange(n_features), data.columns.values)
#     plt.xlabel('Feature Importance')
#     plt.ylabel('Feature')
    
def feature_plot(transformer, gridsearch, X):
    # getting the matrix
    transformer.transform(X)

    # getting the list of features
    features = list(gridsearch.best_estimator_[0].transformers_[0][1].get_feature_names())+list(X.select_dtypes('number').columns)

    # getting importances from my gridsearchCV pipeline
    importances = gridsearch.best_estimator_[1].feature_importances_

    # merging  and sorting
    sorted_importances = sorted(list(zip(features, importances)),key=lambda x: x[1], reverse=True)[:25]

    #plotting
    x = [val[0] for val in sorted_importances]
    y = [val[1] for val in sorted_importances]
    plt.figure(figsize=(20,6))
    plt.bar(x, y)
    plt.xticks(rotation=90);