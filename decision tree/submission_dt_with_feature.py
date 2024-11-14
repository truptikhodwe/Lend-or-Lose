import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the test data
test_df = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# Load the saved model
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

test_df.drop('LoanID', axis=1, inplace=True)
X_test = test_df[['LoanAmount']]

# Predict the 'Default' column using the loaded model
predictions = loaded_model.predict(X_test)

submission_df = sample_submission.copy()
submission_df['Default'] = predictions

submission_df.to_csv("submission_dt_with_feature.csv", index=False)
print("submission_2.csv file created successfully.")
