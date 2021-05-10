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

def stops_by_race(df):
    # proportion of terry stops by race
    stop_percent = df['Subject Perceived Race'].value_counts('normalize=True')

    sdf = pd.DataFrame(stop_percent)
    sdf.reset_index(inplace=True)
    sdf.columns = ['Race', 'Percentage']

    fig, ax = plt.subplots(figsize=(20,8))
    sns.barplot(x='Percentage', y='Race', edgecolor='deepskyblue', palette='Blues_r', data=sdf)
    ax.tick_params(labelsize=20)
    plt.title('Proportion of Terry Stops by Race', fontsize=30)
    plt.xlabel('', fontsize=20)
    plt.ylabel("")
    # plt.savefig('stops_by_race',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()
    
def outcome_by_race(df):
    # creating an arrest rate data frame and visualization
    arrest_rates = df[df['Target']==1]['Subject Perceived Race'].value_counts(normalize=True)

    adf = pd.DataFrame(arrest_rates)
    adf.reset_index(inplace=True)
    adf.columns = ['Race', 'Proportion of Major Outcomes']

    fig, ax = plt.subplots(figsize =(20, 8))
    sns.barplot(x='Proportion of Major Outcomes', y='Race', data=adf, edgecolor='deepskyblue', palette='Blues_r')
    plt.title('Proportion of Major Outcomes by Race', fontsize=30)
    ax.tick_params(labelsize=20)
    plt.xlabel("")
    plt.ylabel("")
    # plt.savefig('outcomes_by_race',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()

def weapons_rate(df):
    wdf = pd.DataFrame(df.groupby('Subject Perceived Race')['Weapon Flag'].value_counts())
    # calculating the weapons found rate by race
    races = df['Subject Perceived Race'].unique()
    hit_rates = []
    for race in races:
        hits = wdf.loc[race]['Weapon Flag'][1]   #number of people in the racial group found with a weapon
        #total number in the racial group stopped
        total = df[df['Subject Perceived Race']\
                   == race].value_counts().sum() 
        rate = hits/total
        hit_rates.append([race, rate])

    #  calculating the meat weapons found rate
    mean_hit_rate = np.array([hit_rates[i][1] for i in range(len(hit_rates))]).mean()

    # creating a hit rate data frame and visualization
    hr_df = pd.DataFrame(hit_rates, columns = ['Race', 'Hit Rate']).sort_values(by='Hit Rate', ascending=False)

    fig, ax = plt.subplots(figsize =(20, 8))
    sns.barplot(y='Race', x='Hit Rate', data=hr_df, palette='Blues_r', edgecolor='deepskyblue')
    plt.title('Weapons Found Rate by Race', fontsize=30)
    ax.tick_params(labelsize=20)
    plt.ylabel("")
    plt.xlabel('', fontsize=20)
    # plt.savefig('weapons_by_race',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()
    
    return sorted(hit_rates, key=lambda x: x[1])[::-1]

def outcome_by_beat(df):
    bdf = df.groupby('Beat Flag')['Target'].value_counts(normalize=True).unstack()
    bdf.plot(kind='bar', figsize=(20,8), color=['skyblue', 'darkblue'], edgecolor='deepskyblue')

    plt.xticks(rotation=0)
    plt.title('Outcome Rates by Beat Flag', fontsize=30)
    plt.xlabel('')
    plt.xticks(np.arange(2), ['No Beat Listed','Beat Listed'], rotation=0)
    plt.legend(title='Outcome', labels=['Minor', 'Major'])
    # plt.savefig('Beat Flag',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()
    
def outcome_by_call(df):
    df['Initial Call Type'] = df['Initial Call Type'].replace('NA', 0)
    df['Initial Call Type'] = df['Initial Call Type'].map(lambda x: 1 if x!=0 else 0)
    call_bin = df.groupby('Initial Call Type')['Target'].value_counts(normalize=True).unstack()
    call_bin.plot(kind='bar', figsize=(20,8), color=['skyblue', 'darkblue'], edgecolor='deepskyblue')
    plt.xticks(np.arange(2), ['No Call Origin Listed','Call Origin Listed'], rotation=0)
    plt.title('Outcome Rates by Call Type', fontsize=30)
    plt.xlabel('', fontsize=20)
    plt.legend(title='Outcome', labels=['Minor', 'Major'])
    # plt.savefig('call_flag',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()
    
def positive_coef(pipeline, X):
    coeff = pipeline[1].coef_.flatten()

    features = list(pipeline[0].transformers_[0][1].get_feature_names()) +\
                list(X.select_dtypes('number').columns)

    coefficients = pd.DataFrame(coeff, features, columns=['coef'])\
                            .sort_values(by='coef', ascending=False)

    top20_pos_coef = coefficients.head(20)

    top20_pos_coef.plot(kind='bar', figsize=(15,7), color=['skyblue'])
    plt.title('Top 20 Positive Coefficients')                                                 
    plt.xlabel('Feature Name')
    plt.xticks(rotation=90)
    plt.legend(title='Coefficient', labels=['Value'])
    plt.show()

def negative_coef(pipeline, X):
    coeff = pipeline[1].coef_.flatten()

    features = list(pipeline[0].transformers_[0][1].get_feature_names()) +\
                list(X.select_dtypes('number').columns)

    coefficients = pd.DataFrame(coeff, features, columns=['coef'])\
                            .sort_values(by='coef', ascending=False)

    top20_neg_coef = coefficients.tail(20)
    
    top20_neg_coef.plot(kind='bar', figsize=(15,7), color=['skyblue'])
    plt.title('Top 20 Negative Coefficients')                                                 
    plt.xlabel('Feature Name')
    plt.xticks(rotation=90)
    plt.legend(title='Coefficient', labels=['Value'])
    plt.show()


def positive_odds(pipeline, X):
    coeff = pipeline[1].coef_.flatten()

    features = list(pipeline[0].transformers_[0][1].get_feature_names()) +\
                list(X.select_dtypes('number').columns)

    coefficients = pd.DataFrame(coeff, features, columns=['coef'])\
                               .sort_values(by='coef', ascending=False)
    
    odds = np.exp(coeff)
    odds_df = pd.DataFrame(odds, features, columns=['odds'])\
                           .sort_values(by='odds', ascending=False)
    top20_pos_odds = odds_df.head(20).reset_index()
    fig, ax = plt.subplots(figsize =(20, 8))
    sns.barplot(x='index',y='odds', data=top20_pos_odds, palette='Blues_r', edgecolor='deepskyblue')
    plt.title('Relative Odds For The Top 20 Positive Features')                                                 
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=90)
    plt.legend(title='Odds That A Major Outcome Is More Likely', labels=['Multiple'])
    # plt.savefig('Baseline_Positive',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()
    
def negative_odds(pipeline, X):
    coeff = pipeline[1].coef_.flatten()

    features = list(pipeline[0].transformers_[0][1].get_feature_names()) +\
                list(X.select_dtypes('number').columns)

    coefficients = pd.DataFrame(coeff, features, columns=['coef'])\
                               .sort_values(by='coef', ascending=False)
    
    odds = np.exp(coeff)
    odds_df = pd.DataFrame(odds, features, columns=['odds'])\
                           .sort_values(by='odds', ascending=False)
    
    top20_neg_odds = odds_df.tail(20).reset_index()

    top20_neg_odds['odds'] = 1/top20_neg_odds['odds']

    fig, ax = plt.subplots(figsize =(20, 8))
    sns.barplot(x='index',y='odds', data=top20_neg_odds, palette='Blues', edgecolor='deepskyblue')
    plt.title('Relative Odds For Top 20 Negative Features')                                                 
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.legend(title='Odds That A Minor Outcome Is More Likely', labels=['Multiple'])
    # plt.savefig('Baseline_Negative',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()
    
    
def importance_plot(pipeline, X):
    """Returns feature importances of the best estimator of a gridsearch."""
    features = list(pipeline[0].transformers_[0][1].get_feature_names()) +\
                list(X.select_dtypes('number').columns)
    
    importances = pipeline[1].feature_importances_
    sorted_importances = sorted(list(zip(features, importances)),key=lambda x: x[1], reverse=True)[:25]
    x = [val[0] for val in sorted_importances]
    y = [val[1] for val in sorted_importances]
    plt.figure(figsize=(20,6))
    sns.barplot(x=x, y=y, palette='Blues_r', edgecolor='deepskyblue')
    plt.xticks(rotation=90)
#     plt.savefig('Feature_Imp',  bbox_inches ="tight",\
#                 pad_inches = .25, transparent = False)
    plt.show()