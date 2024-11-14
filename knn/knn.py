import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
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

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()
n_components = len(cumulative_variance[cumulative_variance < 0.95]) + 1


pca = PCA(n_components=n_components)
print(pca)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

knn = KNeighborsClassifier()
parameters = {'n_neighbors': [3, 5, 7, 9, 11],
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan']}

grid_search = GridSearchCV(estimator=knn, param_grid=parameters, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy:", best_accuracy)
print("Best Parameters:", best_parameters)

best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy on test set:", accuracy_score(y_test,y_pred))

filename = 'finalized_model.sav'
pickle.dump(best_knn, open(filename, 'wb'))

# Saving the StandardScaler
scaler_filename = 'scaler.sav'
pickle.dump(sc, open(scaler_filename, 'wb'))

#Saving the PCA
pca_filename = 'pca.sav'
pickle.dump(pca, open(pca_filename, 'wb'))
