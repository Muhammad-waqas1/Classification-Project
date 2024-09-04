import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
emails = pd.read_csv('emails.csv')

# Drop the 'Email No.' column
emails = emails.drop(['Email No.'], axis=1)

# Dummy label column
emails['Label'] = np.random.randint(2, size=len(emails))

# Split the dataset into features and target
X = emails.drop(['Label'], axis=1)
y = emails['Label']

# Training and testing of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Logistic Regression model with scaled data
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_scaled)

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
print(f'F1-Score: {f1_score(y_test, y_pred)}')

from sklearn.ensemble import RandomForestClassifier

# Train the Random Forest model
clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)

y_pred_rf = clf_rf.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred_rf)}')
print(f'Precision: {precision_score(y_test, y_pred_rf)}')
print(f'Recall: {recall_score(y_test, y_pred_rf)}')
print(f'F1-Score: {f1_score(y_test, y_pred_rf)}')

from sklearn.model_selection import GridSearchCV

# parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize Grid Search
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=3, scoring='accuracy')

# Fit Grid Search
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Train the model with the best parameters
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# Predict on the test set
y_pred_best_rf = best_rf.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred_best_rf)}')
print(f'Precision: {precision_score(y_test, y_pred_best_rf)}')
print(f'Recall: {recall_score(y_test, y_pred_best_rf)}')
print(f'F1-Score: {f1_score(y_test, y_pred_best_rf)}')