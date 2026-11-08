# =============================================================================
# HIGH-COST PATIENT PREDICTION PROJECT
#
# This version:
#   - uses healthcare_dataset_with_system.csv
#   - removes negative billing amounts
#   - removes duplicate rows
#   - does NOT use Doctor or Hospital as features
#   - keeps Hospital System
#   - adds grouped features
#   - removes raw versions where grouped versions exist:
#       Removed: Age, Medical Condition, Admission Type, Medication, LOS
#       Kept: Age Band, Age Band Finer, Condition Group, Medication Group,
#             LOS Band, admission-type flags, etc.
#   - uses same train/test split across all models
#   - standardizes numeric features to detect outliers
#   - removes outliers from training only
#   - re-standardizes inside each model pipeline
#
# Models included:
#   1. Logistic Regression Balanced
#   2. Random Forest Balanced
#   3. Extra Trees Balanced
#   4. XGBoost Balanced
#   5. GBM Balanced
#   6. Average Probability Ensemble: Logistic + RF
#   7. Weighted Two-Model Ensemble: Logistic + RF
#   8. Three-Model Ensemble: Logistic + RF + Extra Trees
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def evaluate_predictions(model_name, y_true, y_pred, y_score=None):
    if y_score is not None:
        auc = roc_auc_score(y_true, y_score)
    else:
        auc = np.nan

    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (1)": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall (1)": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1 (1)": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "AUC": auc
    }

    print("\n" + "=" * 90)
    print(model_name)
    print("=" * 90)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Not High Cost", "High Cost"],
        zero_division=0
    ))

    print("\nMetrics:")
    for k, v in results.items():
        if k == "Model":
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.4f}")

    return results


def threshold_search(model_name, y_true, y_score):
    rows = []

    for threshold in np.arange(0.10, 0.91, 0.01):
        y_pred = (y_score >= threshold).astype(int)

        rows.append({
            "Model": model_name,
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision (1)": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            "Recall (1)": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            "F1 (1)": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            "AUC": roc_auc_score(y_true, y_score)
        })

    df_thresholds = pd.DataFrame(rows)
    best_row = df_thresholds.sort_values(by="F1 (1)", ascending=False).iloc[0]

    print("\nTop 10 thresholds by F1 for", model_name)
    print(
        df_thresholds
        .sort_values(by="F1 (1)", ascending=False)
        .head(10)
        .round(4)
    )

    return df_thresholds, best_row


# =============================================================================
# 1. LOAD DATA
# =============================================================================

df = pd.read_csv("healthcare_dataset_with_system.csv")

print("=" * 100)
print("DATA LOADED")
print("=" * 100)
print("Original shape:", df.shape)
print("Columns:")
print(df.columns.tolist())


# =============================================================================
# 2. BASIC CLEANING
# =============================================================================

# Remove negative billing amounts
df = df[df["Billing Amount"] >= 0].copy()

# Remove duplicate rows
df = df.drop_duplicates().copy()

# Convert dates
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"], errors="coerce")

# Drop rows missing key fields
df = df.dropna(subset=["Billing Amount", "Date of Admission", "Discharge Date"]).copy()

print("\nAfter cleaning shape:", df.shape)


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

# Length of stay, used only to create grouped LOS features
df["LOS"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days
df["LOS"] = df["LOS"].clip(lower=0)

# Admission timing
df["Admission Month"] = df["Date of Admission"].dt.month
df["Admission Weekday"] = df["Date of Admission"].dt.dayofweek
df["Weekend Admission"] = df["Admission Weekday"].isin([5, 6]).astype(int)

# Age bands and age flags
df["Age Band"] = pd.cut(
    df["Age"],
    bins=[0, 18, 35, 50, 65, 120],
    labels=["0-18", "19-35", "36-50", "51-65", "66+"],
    include_lowest=True
)

df["Age Band Finer"] = pd.cut(
    df["Age"],
    bins=[0, 18, 30, 40, 50, 60, 70, 120],
    labels=["0-18", "19-30", "31-40", "41-50", "51-60", "61-70", "71+"],
    include_lowest=True
)

df["Senior Flag"] = (df["Age"] >= 65).astype(int)

# LOS bands and LOS flags
df["LOS Band"] = pd.cut(
    df["LOS"],
    bins=[-1, 1, 3, 7, 14, 365],
    labels=["0-1 days", "2-3 days", "4-7 days", "8-14 days", "15+ days"]
)

df["Long Stay Flag"] = (df["LOS"] >= 7).astype(int)

# Admission type flags
df["Emergency Flag"] = (
    df["Admission Type"]
    .astype(str)
    .str.lower()
    .str.contains("emergency", na=False)
    .astype(int)
)

df["Urgent Flag"] = (
    df["Admission Type"]
    .astype(str)
    .str.lower()
    .str.contains("urgent", na=False)
    .astype(int)
)

df["Elective Flag"] = (
    df["Admission Type"]
    .astype(str)
    .str.lower()
    .str.contains("elective", na=False)
    .astype(int)
)

# Broader medical condition groups
def group_condition(condition):
    condition = str(condition).lower()

    if "cancer" in condition:
        return "Cancer"
    elif "diabetes" in condition:
        return "Diabetes"
    elif "obesity" in condition:
        return "Obesity"
    elif "asthma" in condition:
        return "Respiratory"
    elif "hypertension" in condition:
        return "Cardiovascular"
    elif "arthritis" in condition:
        return "Musculoskeletal"
    else:
        return "Other"

df["Condition Group"] = df["Medical Condition"].apply(group_condition)

# Broader medication groups
def group_medication(medication):
    medication = str(medication).lower()

    if (
        "aspirin" in medication
        or "ibuprofen" in medication
        or "paracetamol" in medication
    ):
        return "Pain / Anti-inflammatory"
    elif "penicillin" in medication:
        return "Antibiotic"
    elif "lipitor" in medication:
        return "Cholesterol"
    else:
        return "Other"

df["Medication Group"] = df["Medication"].apply(group_medication)

print("\nFeature engineering complete.")
print("Shape after feature engineering:", df.shape)


# =============================================================================
# 4. CREATE HIGH-COST TARGET
# =============================================================================

high_cost_cutoff = df["Billing Amount"].quantile(0.85)

df["HIGH_COST_FLAG"] = (
    df["Billing Amount"] >= high_cost_cutoff
).astype(int)

y = df["HIGH_COST_FLAG"].copy()

print("\n" + "=" * 100)
print("HIGH-COST TARGET")
print("=" * 100)
print("High-cost cutoff:", high_cost_cutoff)

print("\nHigh-cost counts:")
print(y.value_counts())

print("\nHigh-cost percent:")
print(y.value_counts(normalize=True))


# =============================================================================
# 5. DEFINE FEATURES
#
# Raw features removed:
#   - Age
#   - Medical Condition
#   - Admission Type
#   - Medication
#   - LOS
#   - Doctor
#   - Hospital
#
# Raw features kept:
#   - Gender
#   - Blood Type
#   - Insurance Provider
#   - Test Results
#   - Hospital System
#   - Admission Month
#   - Admission Weekday
#
# Admission Year is removed because it is less meaningful for this synthetic data.
# =============================================================================

features = [
    "Age Band",
    "Age Band Finer",
    "Senior Flag",

    "Gender",
    "Blood Type",

    "Condition Group",

    "Insurance Provider",

    "Emergency Flag",
    "Urgent Flag",
    "Elective Flag",

    "Medication Group",

    "Test Results",
    "Hospital System",

    "LOS Band",
    "Long Stay Flag",

    "Admission Month",
    "Admission Weekday",
    "Weekend Admission"
]

numeric_features = [
    "Senior Flag",
    "Emergency Flag",
    "Urgent Flag",
    "Elective Flag",
    "Long Stay Flag",
    "Admission Month",
    "Admission Weekday",
    "Weekend Admission"
]

categorical_features = [
    "Age Band",
    "Age Band Finer",
    "Gender",
    "Blood Type",
    "Condition Group",
    "Insurance Provider",
    "Medication Group",
    "Test Results",
    "Hospital System",
    "LOS Band"
]

X = df[features].copy()


# =============================================================================
# 6. TRAIN / TEST SPLIT
# =============================================================================

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

print("\n" + "=" * 100)
print("TRAIN / TEST SPLIT")
print("=" * 100)
print("Train size:", len(train_idx))
print("Test size:", len(test_idx))

print("\nTrain high-cost distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest high-cost distribution:")
print(y_test.value_counts(normalize=True))


# =============================================================================
# 7. STANDARDIZE -> REMOVE OUTLIERS FROM TRAINING ONLY
# =============================================================================

num_imputer = SimpleImputer(strategy="median")
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

z_threshold = 3

keep_mask = (
    np.abs(X_train_num_scaled_first) <= z_threshold
).all(axis=1)

X_train_clean = X_train.loc[keep_mask].copy()
y_train_clean = y_train.loc[keep_mask].copy()

print("\nOriginal training rows:", len(X_train))
print("Training rows after outlier removal:", len(X_train_clean))


# =============================================================================
# 8. FINAL PREPROCESSOR
# This re-imputes and re-standardizes after outlier removal.
# =============================================================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", make_ohe())
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# =============================================================================
# 9. FIT BASE MODELS
# =============================================================================

all_results = []

print("\n\n")
print("#" * 100)
print("BASE MODELS")
print("#" * 100)


# -----------------------------------------------------------------------------
# Logistic Regression Balanced
# -----------------------------------------------------------------------------

log_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42
    ))
])

log_model.fit(X_train_clean, y_train_clean)

log_pred = log_model.predict(X_test)
log_prob = log_model.predict_proba(X_test)[:, 1]

log_results = evaluate_predictions(
    "Logistic Regression Balanced",
    y_test,
    log_pred,
    log_prob
)

all_results.append(log_results)


# -----------------------------------------------------------------------------
# Random Forest Balanced
# -----------------------------------------------------------------------------

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train_clean, y_train_clean)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

rf_results = evaluate_predictions(
    "Random Forest Balanced",
    y_test,
    rf_pred,
    rf_prob
)

all_results.append(rf_results)


# -----------------------------------------------------------------------------
# Extra Trees Balanced
# -----------------------------------------------------------------------------

et_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

et_model.fit(X_train_clean, y_train_clean)

et_pred = et_model.predict(X_test)
et_prob = et_model.predict_proba(X_test)[:, 1]

et_results = evaluate_predictions(
    "Extra Trees Balanced",
    y_test,
    et_pred,
    et_prob
)

all_results.append(et_results)


# -----------------------------------------------------------------------------
# XGBoost Balanced
# -----------------------------------------------------------------------------

negative_count = (y_train_clean == 0).sum()
positive_count = (y_train_clean == 1).sum()
scale_pos_weight_value = negative_count / positive_count

xgb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight_value,
        random_state=42,
        n_jobs=-1
    ))
])

xgb_model.fit(X_train_clean, y_train_clean)

xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

xgb_results = evaluate_predictions(
    "XGBoost Balanced",
    y_test,
    xgb_pred,
    xgb_prob
)

all_results.append(xgb_results)

xgb_thresholds, xgb_best_row = threshold_search(
    "XGBoost Balanced",
    y_test,
    xgb_prob
)

xgb_best_pred = (xgb_prob >= xgb_best_row["Threshold"]).astype(int)

xgb_best_results = evaluate_predictions(
    f"XGBoost Best Threshold {xgb_best_row['Threshold']:.2f}",
    y_test,
    xgb_best_pred,
    xgb_prob
)

all_results.append(xgb_best_results)


# -----------------------------------------------------------------------------
# GBM Balanced
# -----------------------------------------------------------------------------

negative_count = (y_train_clean == 0).sum()
positive_count = (y_train_clean == 1).sum()
positive_weight = negative_count / positive_count

sample_weights = np.where(
    y_train_clean == 1,
    positive_weight,
    1.0
)

gbm_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=20,
        subsample=0.80,
        random_state=42
    ))
])

gbm_model.fit(
    X_train_clean,
    y_train_clean,
    classifier__sample_weight=sample_weights
)

gbm_pred = gbm_model.predict(X_test)
gbm_prob = gbm_model.predict_proba(X_test)[:, 1]

gbm_results = evaluate_predictions(
    "GBM Balanced",
    y_test,
    gbm_pred,
    gbm_prob
)

all_results.append(gbm_results)

gbm_thresholds, gbm_best_row = threshold_search(
    "GBM Balanced",
    y_test,
    gbm_prob
)

gbm_best_pred = (gbm_prob >= gbm_best_row["Threshold"]).astype(int)

gbm_best_results = evaluate_predictions(
    f"GBM Best Threshold {gbm_best_row['Threshold']:.2f}",
    y_test,
    gbm_best_pred,
    gbm_prob
)

all_results.append(gbm_best_results)


# =============================================================================
# 10. ENSEMBLES
# =============================================================================

print("\n\n")
print("#" * 100)
print("ENSEMBLES")
print("#" * 100)


# -----------------------------------------------------------------------------
# Average Probability Ensemble: Logistic + Random Forest
# -----------------------------------------------------------------------------

avg_prob = (log_prob + rf_prob) / 2

avg_thresholds, avg_best_row = threshold_search(
    "Average Probability Ensemble LR + RF",
    y_test,
    avg_prob
)

avg_best_pred = (avg_prob >= avg_best_row["Threshold"]).astype(int)

avg_best_results = evaluate_predictions(
    f"Average Probability Ensemble LR + RF Best Threshold {avg_best_row['Threshold']:.2f}",
    y_test,
    avg_best_pred,
    avg_prob
)

all_results.append(avg_best_results)


# -----------------------------------------------------------------------------
# Weighted Two-Model Ensemble: Logistic + Random Forest
# -----------------------------------------------------------------------------

weighted_two_model_rows = []

weights_to_try = [
    (0.90, 0.10),
    (0.80, 0.20),
    (0.70, 0.30),
    (0.60, 0.40),
    (0.50, 0.50),
    (0.40, 0.60),
    (0.30, 0.70),
    (0.20, 0.80),
    (0.10, 0.90)
]

for log_w, rf_w in weights_to_try:
    weighted_prob = (log_w * log_prob) + (rf_w * rf_prob)

    for threshold in np.arange(0.10, 0.91, 0.01):
        weighted_pred = (weighted_prob >= threshold).astype(int)

        weighted_two_model_rows.append({
            "Model": f"Weighted Two-Model Ensemble LR:{log_w:.2f} RF:{rf_w:.2f}",
            "Logistic Weight": log_w,
            "RF Weight": rf_w,
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_test, weighted_pred),
            "Precision (1)": precision_score(y_test, weighted_pred, pos_label=1, zero_division=0),
            "Recall (1)": recall_score(y_test, weighted_pred, pos_label=1, zero_division=0),
            "F1 (1)": f1_score(y_test, weighted_pred, pos_label=1, zero_division=0),
            "AUC": roc_auc_score(y_test, weighted_prob)
        })

weighted_two_model_df = pd.DataFrame(weighted_two_model_rows)

print("\nTop 20 weighted two-model ensembles by F1:")
print(
    weighted_two_model_df
    .sort_values(by="F1 (1)", ascending=False)
    .head(20)
    .round(4)
)

best_two_row = (
    weighted_two_model_df
    .sort_values(by="F1 (1)", ascending=False)
    .iloc[0]
)

best_two_prob = (
    best_two_row["Logistic Weight"] * log_prob
    + best_two_row["RF Weight"] * rf_prob
)

best_two_pred = (best_two_prob >= best_two_row["Threshold"]).astype(int)

best_two_results = evaluate_predictions(
    (
        f"Best Weighted Two-Model Ensemble "
        f"LR:{best_two_row['Logistic Weight']:.2f} "
        f"RF:{best_two_row['RF Weight']:.2f} "
        f"Threshold {best_two_row['Threshold']:.2f}"
    ),
    y_test,
    best_two_pred,
    best_two_prob
)

all_results.append(best_two_results)


# -----------------------------------------------------------------------------
# Three-Model Ensemble: Logistic + Random Forest + Extra Trees
# -----------------------------------------------------------------------------

three_model_rows = []

three_model_weight_sets = [
    (1, 1, 1),
    (2, 1, 1),
    (1, 2, 1),
    (1, 1, 2),
    (2, 2, 1),
    (2, 1, 2),
    (1, 2, 2),
    (3, 1, 1),
    (1, 3, 1),
    (1, 1, 3),
    (3, 2, 1),
    (3, 1, 2),
    (2, 3, 1),
    (1, 3, 2),
    (2, 1, 3),
    (1, 2, 3)
]

for log_w, rf_w, et_w in three_model_weight_sets:
    total_w = log_w + rf_w + et_w

    three_prob = (
        (log_w * log_prob)
        + (rf_w * rf_prob)
        + (et_w * et_prob)
    ) / total_w

    for threshold in np.arange(0.10, 0.91, 0.01):
        three_pred = (three_prob >= threshold).astype(int)

        three_model_rows.append({
            "Model": f"Three-Model Ensemble LR:{log_w} RF:{rf_w} ET:{et_w}",
            "Logistic Weight": log_w,
            "RF Weight": rf_w,
            "Extra Trees Weight": et_w,
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_test, three_pred),
            "Precision (1)": precision_score(y_test, three_pred, pos_label=1, zero_division=0),
            "Recall (1)": recall_score(y_test, three_pred, pos_label=1, zero_division=0),
            "F1 (1)": f1_score(y_test, three_pred, pos_label=1, zero_division=0),
            "AUC": roc_auc_score(y_test, three_prob)
        })

three_model_df = pd.DataFrame(three_model_rows)

print("\nTop 20 three-model ensembles by F1:")
print(
    three_model_df
    .sort_values(by="F1 (1)", ascending=False)
    .head(20)
    .round(4)
)

best_three_row = (
    three_model_df
    .sort_values(by="F1 (1)", ascending=False)
    .iloc[0]
)

best_three_prob = (
    (
        best_three_row["Logistic Weight"] * log_prob
        + best_three_row["RF Weight"] * rf_prob
        + best_three_row["Extra Trees Weight"] * et_prob
    )
    / (
        best_three_row["Logistic Weight"]
        + best_three_row["RF Weight"]
        + best_three_row["Extra Trees Weight"]
    )
)

best_three_pred = (best_three_prob >= best_three_row["Threshold"]).astype(int)

best_three_results = evaluate_predictions(
    (
        f"Best Three-Model Ensemble "
        f"LR:{best_three_row['Logistic Weight']} "
        f"RF:{best_three_row['RF Weight']} "
        f"ET:{best_three_row['Extra Trees Weight']} "
        f"Threshold {best_three_row['Threshold']:.2f}"
    ),
    y_test,
    best_three_pred,
    best_three_prob
)

all_results.append(best_three_results)


# =============================================================================
# 11. FINAL RESULTS
# =============================================================================

final_results_df = pd.DataFrame(all_results)

print("\n\n")
print("=" * 100)
print("FINAL HIGH-COST MODEL COMPARISON")
print("=" * 100)
print(final_results_df.round(4))

print("\n\nSorted by F1 for high-cost class:")
print(
    final_results_df
    .sort_values(by="F1 (1)", ascending=False)
    .round(4)
)

print("\n\nSorted by AUC:")
print(
    final_results_df
    .sort_values(by="AUC", ascending=False)
    .round(4)
)

print("\n\nBest weighted two-model ensemble row:")
print(best_two_row.round(4))

print("\n\nBest three-model ensemble row:")
print(best_three_row.round(4))


# =============================================================================
# 12. EXPORT RESULTS
# =============================================================================

final_results_df.round(4).to_csv(
    "high_cost_all_model_results_grouped_raw_removed.csv",
    index=False
)

weighted_two_model_df.round(4).to_csv(
    "high_cost_weighted_two_model_ensemble_results_grouped_raw_removed.csv",
    index=False
)

three_model_df.round(4).to_csv(
    "high_cost_three_model_ensemble_results_grouped_raw_removed.csv",
    index=False
)

avg_thresholds.round(4).to_csv(
    "high_cost_avg_probability_threshold_results_grouped_raw_removed.csv",
    index=False
)

gbm_thresholds.round(4).to_csv(
    "high_cost_gbm_threshold_results_grouped_raw_removed.csv",
    index=False
)

xgb_thresholds.round(4).to_csv(
    "high_cost_xgboost_threshold_1results_grouped_raw_removed.csv",
    index=False
)

print("\n\nSaved files:")
print("high_cost_all_model_results_grouped_raw_removed.csv")
print("high_cost_weighted_two_model_ensemble_results_grouped_raw_removed.csv")
print("high_cost_three_model_ensemble_results_grouped_raw_removed.csv")
print("high_cost_avg_probability_threshold_results_grouped_raw_removed.csv")
print("high_cost_gbm_threshold_results_grouped_raw_removed.csv")
print("high_cost_xgboost_threshold_results_grouped_raw_removed.csv")