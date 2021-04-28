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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, make_scorer
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
        print(scores)
        return scores

    def print_error(self, name, Accuracy):
        print(f'{name} has an average percision of {Accuracy}')

class HarnessCCV:
    
    def __init__(self, scorer, random_state=2021):
        self.scorer = scorer
        self.history = pd.DataFrame(columns=['Name', 'Accuracy', 'Notes'])

    def report(self, estimator, X, y, name, notes=''):
        # Create a list to hold the scores from each fold
        kfold_val_scores = np.ndarray(5)
        kfold_train_scores = np.ndarray(5)

        # Instantiate a splitter object and loop over its result
        kfold = StratifiedKFold(n_splits=5)
        for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
            # Extract train and validation subsets using the provided indices
            X_t, X_val = X.iloc[train_index], X.iloc[val_index]
            y_t, y_val = y.iloc[train_index], y.iloc[val_index]
        
            # Instantiate StandardScaler
            scaler = StandardScaler()
            # Fit and transform X_t
            X_t_scaled = scaler.fit_transform(X_t)
            # Transform X_val
            X_val_scaled = scaler.transform(X_val)
        
            # Instantiate SMOTE
            sm = SMOTE(random_state=2021)
            # Fit and transform X_t_scaled and y_t using sm
            X_t_oversampled, y_t_oversampled = sm.fit_resample(X_t_scaled, y_t)
        
            # Clone the provided model and fit it on the train subset
            temp_model = clone(estimator)
            temp_model.fit(X_t_oversampled, y_t_oversampled)
        
            # Evaluate the model on the validation subsets
            score_train = precision_score(temp_model.predict(X_t_oversampled), y_t_oversampled)
            scores_val = precision_score(temp_model.predict(X_val_scaled), y_val)
            kfold_train_scores[fold] = score_train
            kfold_val_scores[fold] = scores_val
        
        frame = pd.DataFrame([[name, scores_val.mean(), notes]], columns=['Name', 'Accuracy', 'Notes'])
        self.history = self.history.append(frame)
        self.history = self.history.reset_index(drop=True)
        self.history = self.history.sort_values('Accuracy')
        self.print_error(name, scores_val.mean())
        print(kfold_val_scores)
                               
        return kfold_train_scores, kfold_val_scores 

    def print_error(self, name, Accuracy):
        print(f'{name} has an average percision of {Accuracy}')