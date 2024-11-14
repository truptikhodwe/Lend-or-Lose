import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

test_df = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

test_df.drop('LoanID', axis=1, inplace=True)

le = LabelEncoder()
for col in ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose', 'HasMortgage', 'HasDependents', 'HasCoSigner']:
    test_df[col] = le.fit_transform(test_df[col])

X_test = test_df 
predictions = loaded_model.predict(X_test)

submission_df = sample_submission.copy()
submission_df['Default'] = predictions

submission_df.to_csv("submission_dt_without_feature.csv", index=False)
print("submission.csv file created successfully.")
