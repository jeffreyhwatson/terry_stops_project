# Terry Stops Project

**Author:** Jeffrey Hanif Watson

![graph0](./reports/figures/police_car.jpeg)

"Police Car Lights" by davidsonscott15 is licensed with CC BY 2.0. 

To view a copy of this license, visit [creative commons](https://creativecommons.org/licenses/by/2.0/)
***
### Quick Links

1. [Final Report](notebooks/report/report.ipynb)
2. [Presentation Slides](reports/presentation.pdf)
***
### Setup Instructions

To setup the project environment, `cd` into the project folder and run `conda env create --file 
terry_stops.yml` in your terminal. Next, run `conda activate terry_stops`.
***
## Overview

Terry stops are named after the 1968 Supreme Court decision Terry v. Ohio and involve the temporary detention of a person based on a reasonable suspicion of criminal activity. This project's goal was to create and train a predictive model, based on a dataset of Seattle Terry Stops data, for use by legal assistance organizations, law firms, and district attorneys. Stop outcomes were divided into two groups and the model was trained to predict either Minor Outcomes (Field Contact, Citation/Infraction) or Major Outcomes (Arrest, Offense Report, Referred for Prosecution). Data cleaning, eda, modeling, tuning and evaluation were performed and a random forest classifier was chosen as the the final model for the project. Since we want to avoid both false positives and false negatives for this project, an accuracy measure of F1 was employed because it is sensitive to both types of error. Since an F1 score is a mix of both precision and recall (F1=1 means perfect recall and precision), the results are more easily described in terms of recall and precision. A final F1 accuracy score of 0.90 was achieved at the end of the modeling process, and the model's recall score of .92 meant that 92% of major outcomes were correctly classified as major outcomes. Further, its precision score of .87 indicated that 87% of stops classified as major outcomes truly ended in major outcomes. An alternate logistic regression model with an F1 accuracy score of 0.89 (recall=.90, precision=.88) is also available for use if requested. Next steps for the project include implementing a feature selection algorithm, tuning an XGboost classifier, and further investigating the logistic regression model's adherence to the underlying assumptions of logistic regression in an attempt to further quantify the effect of specific features on the model.
***
## Business Understanding
Individuals and entities, such as legal assistance organizations, law firms, and district attorneys, that need to allocate resources based on major interactions with the judicial system can benefit from a model that can predict such interactions based on Terry stop administrative and demographic data.
***
## Data Understanding
The initial data for the project was obtained as a .csv file of 47,213 rows and 23 (4 numeric and 19 categorical) feature columns of Terry Stops Data. After cleaning and feature engineering, the final data set contained 42,589 rows and 21 (9 numeric and 12 categorical) feature columns.

Originally, the model's target feature was created by dividing the `Stop Resolution` feature categories into two groups: No Arrest and Arrest. Upon moving through the modeling process, it was discovered that the structure of the data was not conducive to  predicting outcomes with the target formulated in this manner.

Ultimately, the target feature was re-formulated by grouping the `Stop Resolution` feature categories into two bins: Minor Outcomes (Field Contact, Citation/Infraction') and Major Outcomes (Arrest, Offense Report, Referred for Prosecution). This new target dramatically improved performance across all models.

Data set obtained from:
[data.gov](https://catalog.data.gov/dataset/terry-stops)

Original data columns explanations:
[data.seattle.gov](https://data.seattle.gov/Public-Safety/Terry-Stops/28ny-9ts8)
***
## Data Preparation
Data cleaning details for the project can be found here:
[Data Cleaning Notebook](notebooks/exploratory/cleaning_eda.ipynb)

Light data cleaning was performed to reformat certain strings and rename some categories for convenience. Nulls and placeholder values were replaced by the string `NA`. Call Time And Date features were converted to datetime objects and split. No rows were dropped during the data preparation process.
***
## Exploring the Stop Data (Highlights From the EDA)

EDA for the project is detailed in the following notebooks:

1. [Initial Analysis Notebook (Original Target)](notebooks/exploratory/eda_visuals.ipynb)
2. [Final Analysis Notebook (Reformulated Target)](notebooks/exploratory/remixed_eda_visuals.ipynb)

#### General Information

- 59% of Stops End in a Major Outcome
- 45% of Stops Originate from 911 Calls & 19% Are Initiated by Officer Observations. 
- 82.7% of 911 originated stops end in a major outcome.
- 66.6% of officer initiated stops end in a major outcome.
- 11.2% of stops with no origination information end in a major outcome.

#### Proportion of Minor & Major Outcomes By Race
![graph14](./reports/figures/minor_major.png)

59.3% of stops end in a major outcome (Arrest, Offense Report, Referred for Prosecution).

#### Proportion of Terry Stops & Major Outcomes By Race
![graph1](./reports/figures/stops_by_race.png)

![graph2](./reports/figures/outcomes_by_race.png)

#### Disproportionate Outcomes for Certain Racial Groups
When we compare the proportion of stops and major outcomes to the proportion of each racial group in the population, we can see that there are the amount of Terry stops and major outcomes (Arrest, Offense Report, Referred for Prosecution) are disproportionate for certain groups.

**According to July 1, 2019 US Census Data for Seattle and the calculations above:**

- **Asian**: 15.4% of population, 3.2% of stops, 3.2% of major outcomes.

- **White**: (Non-Hispanic) 63.8% of population, 48.9% of stops, 48.3% of major outcomes.

- **Black**: 7.3% of population, 29.9% of stops, 32.1% of major outcomes.

- **Native American**: 0.5% of population, 2.9% of stops, 3.1% of major outcomes.

- **Hispanic**: 6.7% of population, 3.5% of stops, 3.8% of major outcomes.

- **Multi-Racial**: 6.9% of population, 1.7% of stops, 1.8% of major outcomes.

- **Pacific Islander**: 0.3% of population, .12% of stops, .11% of major outcomes.

The Unknown racial category is absent from the census data but makes up 5.6% of stops and 5.1% of major outcomes.

#### Weapon Found Rate By Race

Checking to see if the disparate rate of stops and major outcomes is justified by an increased likelihood of finding a weapon during stops of individuals of certain groups.

![graph3](./reports/figures/weapons_by_race.png)

From the visualization above, it appears that White, Hispanic, and Asian subjects are stopped less frequently than suggested by the weapon found rate. Conversely, Black subjects are stopped elevated rates relative to the weapon found rate.

#### Missing Data Correlated with Minor Outcomes 
Certain missing administrative data is correlated with minor outcomes and figures heavily in the final model. 

For example:

![graph4](./reports/figures/beat_flag.png)

![graph5](./reports/figures/call_flag.png)

- Calls With No Origination Information (28% of calls) Are Much Less Likely to End in a Major Outcome

- Stops With No Beat Information (21% of Stops) Are Much Less Likely to End in A Major Outcome

- Stops With No Precinct Information (21% of Stops) Also Have Much Lower Major Outcome Rates
***
## Modeling

### Baseline Logistic Regression

A baseline logistic regression model was developed and trained on an initial data frame of 46,960 rows and 12 feature columns. Categorical data was one-hot encoded and numerical data was min-max scaled.

![graph6](./reports/figures/Baseline_CM.png)

### Baseline Scores: F1 = 0.87, Recall = .88, Precision = .87

#### Score Interpretation
Since we want to avoid both false positives and false negatives for this project, an accuracy measure of F1 was employed because it is sensitive to both types of error. Also, because F1 is a mix of both precision and recall, the interpretation of the results is more easily described in terms of recall and precision. 

- A recall score of .88 means that 88% of major outcomes were correctly classified as major outcomes. 
- A precision score of .87 indicates that 87% of stops classified as ending in a major outcome truly ended in a major outcome.

#### Baseline Relative Odds

![graph7](./reports/figures/Baseline_Positive.png)

![graph8](./reports/figures/Baseline_Negative.png)

#### Interpretation of the Odds
If the assumptions of logistic regression were met by the model, we could numerically quantify the effect of each feature on the model. However, since it is beyond the scope of the project to ensure that the model meets the underlying assumptions of logistic regression, what we can say about the features above are their relative importances to the model. A higher bar means more importance of the feature to the model. 

### Feature Engineering & Intermediate Models

After a baseline data frame and model were established, additional features were added and/or engineered and tested one by one. Feature engineering techniques included binarizing, binning, and creating new features by combining two base features. For example, an `Officer Age` feature was created by combining the `Officer YOB` (Year of Birth) column with the `Reported Year` Column. After a first round of features were added, models were tuned and tested on different combinations of the new data and a random forest classifier was found to have the greatest F1 accuracy. A second round of feature engineering was undertaken to try and wring a little more accuracy from the models. For example, a simplified `Weapon Bins` was created from the `Weapon Type` column by binning the major weapon types into separate bins and grouping the minor weapon types into a single `Other` category. Also, Rows with missing subject race, subject age, and officer race data were dropped. The models were again tested and tuned on the new data, and the results are presented below.

### Final Logistic Regression Model
A final logistic regression model was developed and trained on a final data frame of 42,589 rows and 21 feature columns. Categorical data was one-hot encoded and numerical data was min-max scaled.

![graph9](./reports/figures/LR_Test_CM.png)

### Final Logistic Regression Scores: F1=0.89, Recall = .90, Precision = .88

- A recall score of .90 means that 90% of major outcomes were correctly classified as major outcomes. 
- A precision score of .88 indicates that 88% of stops classified as ending in a major outcome truly ended in a major outcome.

![graph10](./reports/figures/LR_Final_Positive.png)

![graph11](./reports/figures/LR_Final_Negative.png)

#### Relevant Features and Interpretation of Odds
`Weapon Type` (x1), `Initial Call Type` (x4), and `Officer Squad` (x6) are the only base feature columns that are driving the final logistic regression model. 

Again, if the assumptions of logistic regression were met by the model, we could numerically quantify the effect of each feature on the model. However, with the visualization above, we can point to the relative importances of the features: a higher the bar means more importance of the feature to the model.

### Final Random Forest Classifier

A random forest classifier was trained and tuned on the same data.

![graph12](./reports/figures/RF_Final_CM.png)

### Final Random Forest Scores: F1=.90, Recall = .92, Precision = .87

- A recall score of .92 means that 92% of major outcomes were correctly classified as major outcomes.
- A precision score of .87 indicates that 87% of stops classified as ending in a major outcome truly ended in a major outcome.

#### Feature Importances
![graph13](./reports/figures/RF_Final_Feature_Imp.png)

As noted in the EDA section of the report, data categories of `NA` are the most important factor driving the final model. Of the first 10 features (ranked in terms of importance to the model), only 2 (`Weapon Type` = `x1_None` & `Call Type`= `x5_911`) do not involve data categories of `NA`. Even the `Call Type Bins` = `x9_OTHER` category is made up of a majority of `NA` values. Interestingly, the only two demographic categories that are present in the top features are `Officer Age` and `Subject Perceived Race`.
***
## Conclusion
A random forest classifier with a F1 accuracy score of 0.90 (recall=.92, precision=.87) was attained at the end of the modeling process and chosen as the final model of the project. The recall score of .92 meant that 92% of major outcomes were correctly classified as major outcomes, and the precision score of .87 indicated that 87% of stops classified as ending in a major outcome truly ended in a major outcome. An alternate logistic regression model with an F1 accuracy score of 0.889 (recall=.90, precision=.88) is also available for use by interested parties.
***
## Next Steps

Next steps for the project include:
- Implementing a feature selection algorithm. 
- Tuning an XGBoost classifier. 
- Further investigating the logistic regression model's adherence to the underlying assumptions of logistic regression.
***

## For More Information

Please review our full analysis in [our Jupyter Notebook](./notebooks/report/report.ipynb) or our [presentation](./reports/presentation.pdf).

For any additional questions, please contact **Jeffrey Hanif Watson jeffrey.h.watson@protonmail.com**

## Repository Structure

```
├── README.md
├── data
│   ├── processed
│   └── raw
├── models
├── notebooks
│   ├── exploratory
│   └── report
├── reports
│   └── figures   
└── src
```
