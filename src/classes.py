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
    
class Harness:
    
    def __init__(self, scorer, random_state=2021):
        self.scorer = scorer
        self.history = pd.DataFrame(columns=['Name', 'Accuracy', 'Notes'])

    def report(self, model, X, y, name, notes='', cv=5,):
        scores = cross_val_score(model, X, y, 
                                 scoring=self.scorer, cv=cv)
        frame = pd.DataFrame([[name, scores.mean(), notes]], columns=['Name', 'Accuracy', 'Notes'])
        self.history = self.history.append(frame)
        self.history = self.history.reset_index(drop=True)
        self.history = self.history.sort_values('Accuracy')
        self.print_error(name, scores.mean())
#         print(scores)
        return scores

    def print_error(self, name, Accuracy):
        print(f'{name} has an average F1 of {Accuracy}')



