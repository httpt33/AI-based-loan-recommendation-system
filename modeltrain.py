#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier

#LOADING THE DATA
loan_train = pd.read_csv('train_csv.csv')
print(loan_train.shape)
print(loan_train.head())

#DATA PREPROCESSING
# Filling missing values in 'Gender' with the mode
loan_train['Gender'] = loan_train['Gender'].fillna(loan_train['Gender'].dropna().mode().values[0])

# Filling missing values in 'Married' with the mode
loan_train['Married'] = loan_train['Married'].fillna(loan_train['Married'].dropna().mode().values[0])

# Filling missing values in 'Dependents' with the mode
loan_train['Dependents'] = loan_train['Dependents'].fillna(loan_train['Dependents'].dropna().mode().values[0])

# Filling missing values in 'Self_Employed' with the mode
loan_train['Self_Employed'] = loan_train['Self_Employed'].fillna(loan_train['Self_Employed'].dropna().mode().values[0])

# Filling missing values in 'LoanAmount' with the mean
loan_train['LoanAmount'] = loan_train['LoanAmount'].fillna(loan_train['LoanAmount'].dropna().mean())

# Filling missing values in 'Loan_Amount_Term' with the mode
loan_train['Loan_Amount_Term'] = loan_train['Loan_Amount_Term'].fillna(loan_train['Loan_Amount_Term'].dropna().mode().values[0])

# Filling missing values in 'Credit_History' with the mode
loan_train['Credit_History'] = loan_train['Credit_History'].fillna(loan_train['Credit_History'].dropna().mode().values[0])

print(set(loan_train['Gender'].values.tolist()))
print(set(loan_train['Dependents'].values.tolist()))
print(set(loan_train['Married'].values.tolist()))
print(set(loan_train['Education'].values.tolist()))
print(set(loan_train['Self_Employed'].values.tolist()))
print(set(loan_train['Loan_Status'].values.tolist()))
print(set(loan_train['Property_Area'].values.tolist()))


