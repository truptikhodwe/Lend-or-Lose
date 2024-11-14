import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

test_df = pd.read_csv("test.csv")

model = pickle.load(open('finalized_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))
pca = pickle.load(open('pca.sav', 'rb'))

test_df.drop_duplicates(inplace=True)  
test_df.drop('LoanID', axis=1, inplace=True)  

le = LabelEncoder()
for col in ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose', 'HasMortgage', 'HasDependents', 'HasCoSigner']:
    test_df[col] = le.fit_transform(test_df[col])

X_test = test_df

X_test = scaler.transform(X_test)

X_test = pca.transform(X_test)

predictions = model.predict(X_test)

submission_df = pd.read_csv("sample_submission.csv")
submission_df['Default'] = predictions

submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
