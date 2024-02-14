#IMPORTING LIBRARIES
import joblib
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
loan_train = pd.read_csv('loan_approval_dataset.csv')
print(loan_train.shape)
print(loan_train.head())


#DATA PREPROCESSING
# Filling missing values in 'Education' with the mode
loan_train['Education'] = loan_train['Education'].fillna(loan_train['Education'].dropna().mode().values[0])

# Filling missing values in 'Dependents' with the mode
loan_train['Dependents'] = loan_train['Dependents'].fillna(loan_train['Dependents'].dropna().mode().values[0])

# Filling missing values in 'Self_Employed' with the mode
loan_train['Self_Employed'] = loan_train['Self_Employed'].fillna(loan_train['Self_Employed'].dropna().mode().values[0])

# Filling missing values in 'LoanAmount' with the mean
loan_train['Loan_Amount'] = loan_train['Loan_Amount'].fillna(loan_train['Loan_Amount'].dropna().mean())

# Filling missing values in 'Loan_Amount_Term' with the mode
loan_train['Loan_Amount_Term'] = loan_train['Loan_Amount_Term'].fillna(loan_train['Loan_Amount_Term'].dropna().mode().values[0])

# Filling missing values in 'cibil_score' with the mode
loan_train['Cibil_Score'] = loan_train['Cibil_Score'].fillna(loan_train['Cibil_Score'].dropna().mode().values[0])

'''print(set(loan_train['Dependents'].values.tolist()))
print(set(loan_train['Education'].values.tolist()))
print(set(loan_train['Self_Employed'].values.tolist()))
print(set(loan_train['Loan_Status'].values.tolist()))
print(set(loan_train['Residential_Assets_Value'].values.tolist()))'''

#CONVERTING CATEGORICAL VARIABLES INTO A FORMAT SUITABLE FOR ML MODELS
from sklearn.preprocessing import StandardScaler

# Convert 'Loan_Status' to binary (0 or 1)
loan_train['Loan_Status'] = loan_train['Loan_Status'].map({' Rejected': 0, ' Approved': 1}).astype(int)

# One-hot encoding categorical columns
loan_train = pd.get_dummies(loan_train, columns=['Education', 'Self_Employed', 'Dependents'])

# Standardize numerical columns
standardScaler = StandardScaler()
columns_to_scale = ['Income_Annum', 'Loan_Amount', 'Loan_Amount_Term', 'Residential_Assets_Value', 'Commercial_Assets_Value', 'Luxury_Assets_Value', 'Bank_Asset_Value']
loan_train[columns_to_scale] = standardScaler.fit_transform(loan_train[columns_to_scale])

#CREATING TRAIN AND TEST DATASET
y = loan_train['Loan_Status']
X = loan_train.drop(['Loan_Status', 'Loan_ID'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#X_train.shape, X_test.shape: (491, 20), (123, 20)

from sklearn.model_selection import GridSearchCV

# Define the parameter grid for GridSearchCV
rf_param_grid = {
    'n_estimators': range(1, 1000, 100)
}

# Create a RandomForestClassifier
rf = RandomForestClassifier()

# Create a GridSearchCV object
rf_grid = GridSearchCV(
    estimator=rf,
    param_grid=rf_param_grid,
    scoring="accuracy",
    verbose=0,
    cv=5  # Number of cross-validation folds
)

joblib.dump(rf_grid, "filename", compress=3)

'''# Fit the GridSearchCV object to the training data
rf_grid.fit(X_train, y_train)

# Get the best parameters
best_params = rf_grid.best_params_
print(f'Best parameters: {best_params}')

# Make predictions on the test set
y_pred = rf_grid.predict(X_test)

# Calculate and print the accuracy
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f'Accuracy: {accuracy}')

def feature_imp(df, model):
    feat = pd.DataFrame(columns=['feature', 'importance'])
    feat["feature"] = df.columns
    feat["importance"] = model.best_estimator_.feature_importances_
    return feat.sort_values(by="importance", ascending=False)

# Assuming rf_grid is the trained Random Forest model
feature_importance_rf = feature_imp(X_train, rf_grid)

# Plotting the feature importance for Random Forest
feature_importance_rf.plot("feature", "importance", "barh", figsize=(10, 7), legend=False)
plt.title("Feature Importance for Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()'''