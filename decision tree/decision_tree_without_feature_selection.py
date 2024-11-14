import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

train_df=pd.read_csv("train.csv") 
train_df
train_df.drop_duplicates(inplace=True)  
print(train_df.isnull() .sum()) 
train_df.drop('LoanID', axis=1, inplace=True)     

print(len(pd.unique(train_df['Education'])))
print(len(pd.unique(train_df['EmploymentType'])))
print(len(pd.unique(train_df['MaritalStatus'])))
print(len(pd.unique(train_df['LoanPurpose'])))

le = LabelEncoder()
for col in ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose', 'HasMortgage', 'HasDependents', 'HasCoSigner']:
    train_df[col] = le.fit_transform(train_df[col])

train_df.head()

X = train_df.drop('Default', axis=1)
y = train_df['Default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

tree_classifier = DecisionTreeClassifier(random_state=7)

grid_search = GridSearchCV(estimator=tree_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

best_tree_classifier = grid_search.best_estimator_
y_pred = best_tree_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")

filename = 'finalized_model.sav'
pickle.dump(best_tree_classifier, open(filename, 'wb'))
