import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
df = pd.read_csv("healthcare_dataset_with_system.csv")

# --------------------------------------------------
# 2. Basic cleaning
# --------------------------------------------------
# Remove negative billing amounts
df = df[df['Billing Amount'] >= 0].copy()

# Convert dates
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])

# --------------------------------------------------
# 3. Feature engineering
# --------------------------------------------------
df['LOS'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
df['Admission Month'] = df['Date of Admission'].dt.month
df['Admission Year'] = df['Date of Admission'].dt.year
df['Admission Weekday'] = df['Date of Admission'].dt.dayofweek

# --------------------------------------------------
# 4. Create shared targets
# --------------------------------------------------
y_reg = df['Billing Amount'].copy()

df['COST_CATEGORY'] = pd.qcut(
    df['Billing Amount'],
    q=3,
    labels=[0, 1, 2]
).astype(int)

y_class = df['COST_CATEGORY'].copy()

print("Cost category counts:")
print(df['COST_CATEGORY'].value_counts().sort_index())
print()

# --------------------------------------------------
# 5. Define predictors
# Billing Amount is NOT used as a feature
# --------------------------------------------------
features = [
    'Age',
    'Gender',
    'Blood Type',
    'Medical Condition',
    'Insurance Provider',
    'Admission Type',
    'Medication',
    'Test Results',
    'Hospital System',
    'LOS',
    'Admission Month',
    'Admission Year',
    'Admission Weekday'
]

X = df[features].copy()

numeric_features = [
    'Age',
    'LOS',
    'Admission Month',
    'Admission Year',
    'Admission Weekday'
]

categorical_features = [
    'Gender',
    'Blood Type',
    'Medical Condition',
    'Insurance Provider',
    'Admission Type',
    'Medication',
    'Test Results',
    'Hospital System'
]

# --------------------------------------------------
# 6. Shared train/test split
# --------------------------------------------------
train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=y_class
)

X_train = X.loc[train_idx].copy()
X_test = X.loc[test_idx].copy()

y_train_reg = y_reg.loc[train_idx].copy()
y_test_reg = y_reg.loc[test_idx].copy()

y_train_class = y_class.loc[train_idx].copy()
y_test_class = y_class.loc[test_idx].copy()

print("Train size:", len(train_idx))
print("Test size:", len(test_idx))
print()

# --------------------------------------------------
# 7. Standardize -> remove outliers -> re-standardize
# --------------------------------------------------
num_imputer = SimpleImputer(strategy='median')
first_scaler = StandardScaler()

X_train_num_imputed = pd.DataFrame(
    num_imputer.fit_transform(X_train[numeric_features]),
    columns=numeric_features,
    index=X_train.index
)

X_train_num_scaled_first = pd.DataFrame(
    first_scaler.fit_transform(X_train_num_imputed),
    columns=numeric_features,
    index=X_train.index
)

# Remove outliers from training set only
z_threshold = 3
keep_mask = (np.abs(X_train_num_scaled_first) <= z_threshold).all(axis=1)

X_train_clean = X_train.loc[keep_mask].copy()
y_train_class_clean = y_train_class.loc[keep_mask].copy()

print("Original training rows:", len(X_train))
print("Training rows after outlier removal:", len(X_train_clean))
print()

# Final preprocessing: re-impute and re-standardize
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# --------------------------------------------------
# 8. Multinomial logistic regression model
# --------------------------------------------------
multi_log_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        max_iter=3000,
        multi_class='multinomial',
        solver='lbfgs',
        random_state=42
    ))
])

# --------------------------------------------------
# 9. Fit model
# --------------------------------------------------
multi_log_model.fit(X_train_clean, y_train_class_clean)

# --------------------------------------------------
# 10. Predict
# --------------------------------------------------
y_pred = multi_log_model.predict(X_test)
y_prob = multi_log_model.predict_proba(X_test)

# --------------------------------------------------
# 11. Evaluate
# --------------------------------------------------
print("Multinomial Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test_class, y_pred))
print("F1 Macro:", f1_score(y_test_class, y_pred, average='macro'))
print("F1 Weighted:", f1_score(y_test_class, y_pred, average='weighted'))
print()

print("Confusion Matrix:")
print(confusion_matrix(y_test_class, y_pred))
print()

print("Classification Report:")
print(classification_report(
    y_test_class,
    y_pred,
    target_names=['Low', 'Medium', 'High']
))