import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.stats import skew
import numpy as np
from imblearn.over_sampling import SMOTE

#Loading the training dataset
train_df = pd.read_csv("train.csv")

# Removing duplicates and dropping unnecessary columns
train_df.drop_duplicates(inplace=True)
train_df.drop('LoanID', axis=1, inplace=True)

#Label encoding categorical columns
le = LabelEncoder()
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose', 'HasMortgage', 'HasDependents', 'HasCoSigner']
for col in categorical_cols:
    train_df[col] = le.fit_transform(train_df[col])

#Separating features and target
X = train_df.drop('Default', axis=1)
y = train_df['Default']

#Identify numerical columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

#Applying mild skew reduction to all numerical columns
for col in numerical_cols:
    X[col] = X[col].apply(lambda x: np.log1p(x) if x > 0 else 0)

#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

#Apply SMOTE to handle class imbalance
print("Class distribution before SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=7)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:")
print(y_train.value_counts())

#Training an SVM model
svc = SVC()
print("Default Parameters of SVC:")
print(svc.get_params())

svc.fit(X_train, y_train)

#Predicting and evaluate
y_pred = svc.predict(X_test)
print('Model accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

#Load test dataset and sample submission
test_df = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

#Same pre-processing steps for the testing 
test_df.drop('LoanID', axis=1, inplace=True)
for col in categorical_cols:
    test_df[col] = le.fit_transform(test_df[col])

for col in numerical_cols:
    test_df[col] = test_df[col].apply(lambda x: np.log1p(x) if x > 0 else 0)

X_test = test_df
predictions = svc.predict(X_test)

# Prepare submission file
submission_df = sample_submission.copy()
submission_df['Default'] = predictions
submission_df.to_csv("submission_svm_default_with_smote.csv", index=False)