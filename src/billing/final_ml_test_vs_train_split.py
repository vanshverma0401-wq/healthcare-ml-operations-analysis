import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. LOAD DATA  ← THIS WAS MISSING
# --------------------------------------------------
df = pd.read_csv("healthcare_dataset_with_system.csv")

# --------------------------------------------------
# 2. CLEAN DATA
# --------------------------------------------------
df = df[df['Billing Amount'] >= 0].copy()

# Convert dates
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])

# --------------------------------------------------
# 3. FEATURE ENGINEERING
# --------------------------------------------------
df['LOS'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
df['Admission Month'] = df['Date of Admission'].dt.month
df['Admission Year'] = df['Date of Admission'].dt.year
df['Admission Weekday'] = df['Date of Admission'].dt.dayofweek

# --------------------------------------------------
# 4. DEFINE FEATURES
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

# --------------------------------------------------
# 5. CREATE TARGETS
# --------------------------------------------------
y_reg = df['Billing Amount']

df['COST_CATEGORY'] = pd.qcut(
    df['Billing Amount'],
    q=3,
    labels=[0, 1, 2]
).astype(int)

y_class = df['COST_CATEGORY']

# --------------------------------------------------
# 6. FEATURES (NO BILLING AMOUNT)
# --------------------------------------------------
X = df[features].copy()

# --------------------------------------------------
# 7. SHARED SPLIT
# --------------------------------------------------
train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=y_class
)

X_train = X.loc[train_idx]
X_test = X.loc[test_idx]

y_train_reg = y_reg.loc[train_idx]
y_test_reg = y_reg.loc[test_idx]

y_train_class = y_class.loc[train_idx]
y_test_class = y_class.loc[test_idx]

print("Train size:", len(train_idx))
print("Test size:", len(test_idx))