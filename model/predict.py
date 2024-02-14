import joblib
import sklearn
import numpy
import pandas

def preprocesspersonal(loan_train):
    # One-hot encoding categorical columns
    loan_train = pd.get_dummies(loan_train, columns=['Education', 'Self_Employed', 'Dependents'])

    # Standardize numerical columns
    standardScaler = StandardScaler()
    columns_to_scale = ['Income_Annum', 'Loan_Amount', 'Loan_Amount_Term', 'Residential_Assets_Value', 'Commercial_Assets_Value', 'Luxury_Assets_Value', 'Bank_Asset_Value']
    loan_train[columns_to_scale] = standardScaler.fit_transform(loan_train[columns_to_scale])
    return loan_train
    

def preprocessedu(loan_train):
    # One-hot encoding categorical columns
    loan_train = pd.get_dummies(loan_train, columns=['Gender', 'Dependents', 'Married', 'Education', 'Self_Employed', 'Property_Area'])

    # Standardize numerical columns
    standardScaler = StandardScaler()
    columns_to_scale = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    loan_train[columns_to_scale] = standardScaler.fit_transform(loan_train[columns_to_scale])
    return loan_train

def load_home_model():
 return joblib.load("./model/final_model_edu.joblib")

def load_personal_model():
 return None #joblib.load("./model/final_model_personal.joblib") 

def can_get_(model,data):
    return model.predict(data)



