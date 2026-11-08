import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
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
df = df[df['Billing Amount'] >= 0].copy()

df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])

# --------------------------------------------------
# 3. Feature engineering
# --------------------------------------------------
df['LOS'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
df['Admission Month'] = df['Date of Admission'].dt.month
df['Admission Year'] = df['Date of Admission'].dt.year
df['Admission Weekday'] = df['Date of Admission'].dt.dayofweek

df['Age_LOS'] = df['Age'] * df['LOS']
df['LOS_squared'] = df['LOS'] ** 2
df['Age_squared'] = df['Age'] ** 2
df['LOS_month_interaction'] = df['LOS'] * df['Admission Month']
df['Age_month_interaction'] = df['Age'] * df['Admission Month']

# --------------------------------------------------
# 4. Create target
# --------------------------------------------------
df['COST_CATEGORY'] = pd.qcut(
    df['Billing Amount'],
    q=3,
    labels=[0, 1, 2]
).astype(int)

print("Cost category counts:")
print(df['COST_CATEGORY'].value_counts().sort_index())
print()

# --------------------------------------------------
# 5. Define predictors
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
    'Admission Weekday',
    'Age_LOS',
    'LOS_squared',
    'Age_squared',
    'LOS_month_interaction',
    'Age_month_interaction'
]

X = df[features].copy()
y = df['COST_CATEGORY'].copy()

numeric_features = [
    'Age',
    'LOS',
    'Admission Month',
    'Admission Year',
    'Admission Weekday',
    'Age_LOS',
    'LOS_squared',
    'Age_squared',
    'LOS_month_interaction',
    'Age_month_interaction'
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
# 6. SAME split
# --------------------------------------------------
train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train = X.loc[train_idx].copy()
X_test = X.loc[test_idx].copy()

y_train = y.loc[train_idx].copy()
y_test = y.loc[test_idx].copy()

print("Train size:", len(train_idx))
print("Test size:", len(test_idx))
print()

# --------------------------------------------------
# 7. Outlier removal on training only
# --------------------------------------------------
num_imputer = SimpleImputer(strategy='median')

X_train_num_imputed = pd.DataFrame(
    num_imputer.fit_transform(X_train[numeric_features]),
    columns=numeric_features,
    index=X_train.index
)

z_scores = (X_train_num_imputed - X_train_num_imputed.mean()) / X_train_num_imputed.std(ddof=0)
z_threshold = 3
keep_mask = (np.abs(z_scores) <= z_threshold).all(axis=1)

X_train_clean = X_train.loc[keep_mask].copy()
y_train_clean = y_train.loc[keep_mask].copy()

print("Original training rows:", len(X_train))
print("Training rows after outlier removal:", len(X_train_clean))
print()

# --------------------------------------------------
# 8. Preprocessing
# --------------------------------------------------
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

gbm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# --------------------------------------------------
# 9. Focused tuning around current best region
# --------------------------------------------------
param_dist = {
    'classifier__learning_rate': [0.12, 0.15, 0.18, 0.20],
    'classifier__max_iter': [400, 500, 600, 800],
    'classifier__max_depth': [10, 12, 14, None],
    'classifier__max_leaf_nodes': [63, 127, 255],
    'classifier__min_samples_leaf': [10, 20, 30, 40],
    'classifier__l2_regularization': [0.0, 0.001, 0.01, 0.1],
    'classifier__max_bins': [32, 64, 128]
}

search = RandomizedSearchCV(
    estimator=gbm_pipeline,
    param_distributions=param_dist,
    n_iter=40,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

search.fit(X_train_clean, y_train_clean)

best_gbm = search.best_estimator_

print("Best params:", search.best_params_)
print("Best CV score:", search.best_score_)
print()

# --------------------------------------------------
# 10. Predict
# --------------------------------------------------
y_pred = best_gbm.predict(X_test)

# --------------------------------------------------
# 11. Evaluate
# --------------------------------------------------
print("Further-Tuned GBM Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Macro:", f1_score(y_test, y_pred, average='macro'))
print("F1 Weighted:", f1_score(y_test, y_pred, average='weighted'))
print()

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()

print("Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=['Low', 'Medium', 'High']
))