import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Load the dataset
df = pd.read_csv('emails.csv')

# Display the first few rows of the dataset
print(df.head())

# Drop non-numeric columns
df = df.drop(columns=['Email No.'])

# Separate numeric and non-numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns
non_numeric_cols = df.select_dtypes(exclude=['number']).columns

# Handle missing values in numeric columns
imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

#Assume all non-numeric columns are categorical
if not non_numeric_cols.empty:
    for col in non_numeric_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Separate features and target variable
X = df.drop('Prediction', axis=1)
y = df['Prediction']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Model Selection
# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Dictionary to store model performance
model_performance = {}

# Train and evaluate each model
for name, model in models.items():

    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Cross-validation score
    cv_score = np.mean(cross_val_score(model, X_train, y_train, cv=5))
    
    # Save the performance metrics
    model_performance[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Cross-Validation Score': cv_score
    }
    
    print(f"Model: {name}")
    print(classification_report(y_test, y_pred))
    print(f"Cross-Validation Score: {cv_score}\n")
    
# Display model performance
performance_df = pd.DataFrame(model_performance).T
print(performance_df)

# Step 5: Hyperparameter Tuning with Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search for Random Forest
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")

# Retrain Random Forest with the best parameters
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)
y_pred_best_rf = best_rf.predict(X_test)

# Final evaluation of the tuned model
print("Final Model Performance after Hyperparameter Tuning:")
print(classification_report(y_test, y_pred_best_rf))