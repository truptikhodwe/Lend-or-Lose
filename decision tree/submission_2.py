import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the test data
test_df = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# Load the saved model
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Preprocess the test data similar to training data
# Dropping 'LoanID' as it was dropped in training data
test_df.drop('LoanID', axis=1, inplace=True)

# Encode categorical columns if they are required for 'LoanAmount' (if any)
# Since feature selection chose only 'LoanAmount', encoding isn't needed here.
# You can skip this loop if 'LoanAmount' is numeric in test data.

# Select only the feature 'LoanAmount' for prediction
X_test = test_df[['LoanAmount']]

# Predict the 'Default' column using the loaded model
predictions = loaded_model.predict(X_test)

# Prepare submission file
submission_df = sample_submission.copy()
submission_df['Default'] = predictions

# Save to CSV
submission_df.to_csv("submission_2.csv", index=False)
print("submission_2.csv file created successfully.")
