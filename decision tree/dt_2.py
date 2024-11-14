import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
import pickle

# Load data
train_df = pd.read_csv("train.csv")
train_df.drop_duplicates(inplace=True)
train_df.drop('LoanID', axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose', 'HasMortgage', 'HasDependents', 'HasCoSigner']:
    train_df[col] = le.fit_transform(train_df[col])

# Define features and target
X = train_df.drop('Default', axis=1)
y = train_df['Default']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Initialize the Decision Tree Classifier
tree_classifier = DecisionTreeClassifier(random_state=7)

# Perform Recursive Feature Elimination with Cross-Validation (RFECV) for feature selection
rfecv = RFECV(estimator=tree_classifier, step=1, cv=5, scoring='accuracy')
rfecv.fit(X_train, y_train)

# Select the optimal features
selected_features = X.columns[rfecv.support_]
X_train = rfecv.transform(X_train)
X_test = rfecv.transform(X_test)

print("Selected features:", selected_features)

# Define parameter grid for GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Run GridSearchCV with the selected features
grid_search = GridSearchCV(estimator=tree_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Evaluate the model on the test data
best_tree_classifier = grid_search.best_estimator_
y_pred = best_tree_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")

# Save the trained model
filename = 'finalized_model.sav'
pickle.dump(best_tree_classifier, open(filename, 'wb'))
