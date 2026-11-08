import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions import ConvergenceWarning

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

# --------------------------------------------------
# 4. Define predictors and shared targets
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

# Shared regression target
y_reg = df['Billing Amount'].copy()

# Shared classification target for consistent split across all models
df['COST_CATEGORY_NUM'] = pd.qcut(
    df['Billing Amount'],
    q=3,
    labels=[0, 1, 2]
).astype(int)

y_class = df['COST_CATEGORY_NUM'].copy()

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
# 5. Shared train / test split
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
# 6. First standardization for outlier detection
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

# --------------------------------------------------
# 7. Remove outliers from training set only
# --------------------------------------------------
z_threshold = 3
keep_mask = (np.abs(X_train_num_scaled_first) <= z_threshold).all(axis=1)

X_train_clean = X_train.loc[keep_mask].copy()
y_train_reg_clean = y_train_reg.loc[keep_mask].copy()

print("Original training rows:", len(X_train))
print("Training rows after outlier removal:", len(X_train_clean))
print()

# --------------------------------------------------
# 8. Final preprocessing
# --------------------------------------------------
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
# 9. Lasso pipeline
# --------------------------------------------------
lasso_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Lasso(max_iter=50000, tol=0.01, random_state=42))
])

# --------------------------------------------------
# 10. Tune alpha with cross-validation
# --------------------------------------------------
param_grid = {
    'regressor__alpha': [1, 10, 50, 100, 250, 500]
}

grid_search = GridSearchCV(
    estimator=lasso_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Suppress convergence warnings for cleaner output
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    grid_search.fit(X_train_clean, y_train_reg_clean)

best_lasso = grid_search.best_estimator_

print("Best alpha:", grid_search.best_params_['regressor__alpha'])
print()

# --------------------------------------------------
# 11. Predict
# --------------------------------------------------
y_pred = best_lasso.predict(X_test)

# --------------------------------------------------
# 12. Evaluate
# --------------------------------------------------
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
mae = mean_absolute_error(y_test_reg, y_pred)
r2 = r2_score(y_test_reg, y_pred)

print("Lasso Regression Results")
print("RMSE:", rmse)
print("MAE:", mae)
print("R^2:", r2)
print()

# --------------------------------------------------
# 13. Baseline comparison
# --------------------------------------------------
baseline_pred = np.repeat(y_train_reg_clean.mean(), len(y_test_reg))
baseline_rmse = np.sqrt(mean_squared_error(y_test_reg, baseline_pred))
baseline_mae = mean_absolute_error(y_test_reg, baseline_pred)
baseline_r2 = r2_score(y_test_reg, baseline_pred)

print("Baseline Results")
print("Baseline RMSE:", baseline_rmse)
print("Baseline MAE:", baseline_mae)
print("Baseline R^2:", baseline_r2)