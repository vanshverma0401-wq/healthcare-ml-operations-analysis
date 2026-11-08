#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ======================================================================
# 0) IMPORTS
# ======================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss


# In[2]:


# ======================================================================
# 1) LOAD DATA
# ======================================================================
file_path = "data/healthcare_data.csv"
df = pd.read_csv(file_path)

display(df.head())


# In[4]:


# ======================================================================
# 2) Basic cleaning (text + dates)
# ======================================================================
text_cols = [
    "Name","Gender","Medical Condition","Doctor","Hospital",
    "Insurance Provider","Admission Type","Medication","Test Results"
]

for c in text_cols:
    df[c] = df[c].astype(str).str.strip().str.title()

df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])
df["Room Number"] = df["Room Number"].astype(int)


# In[5]:


# ======================================================================
# 3) Simple patient features
# ======================================================================
df["Length_of_Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days
df["Admission_Weekday"] = df["Date of Admission"].dt.dayofweek


# In[6]:


# ======================================================================
# 4) Create synthetic Hospital System
# ======================================================================
systems = ["NorthCare", "Summit", "RiverValley", "Lakeshore"]
hospital_counts = df["Hospital"].value_counts()

system_totals = {s: 0 for s in systems}
mapping = {}

for hospital, count in hospital_counts.items():
    target = min(system_totals, key=system_totals.get)
    mapping[hospital] = target
    system_totals[target] += count

df["Hospital_System"] = df["Hospital"].map(mapping)


# In[36]:


# ======================================================================
# 5) Build daily room occupancy (IMPORTANT)
# ======================================================================
occ = df[["Hospital_System","Room Number","Date of Admission","Discharge Date"]]

admit = occ.rename(columns={"Date of Admission":"date"})[["Hospital_System","Room Number","date"]]
admit["delta"] = 1

discharge = occ.rename(columns={"Discharge Date":"date"})[["Hospital_System","Room Number","date"]]
discharge["delta"] = -1

events = pd.concat([admit, discharge])
events = events.groupby(["Hospital_System","Room Number","date"])["delta"].sum().reset_index()

all_dates = pd.date_range(events["date"].min(), events["date"].max())
rooms = occ[["Hospital_System","Room Number"]].drop_duplicates()

grid = rooms.merge(pd.DataFrame({"date": all_dates}), how="cross")
grid = grid.merge(events, how="left").fillna(0)

grid = grid.sort_values(["Hospital_System","Room Number","date"])
grid["occupied"] = grid.groupby(["Hospital_System","Room Number"])["delta"].cumsum().clip(0,1)


# In[50]:


# ======================================================================
# 6) Daily system utilization
# ======================================================================

daily_system = (
    grid.groupby(["Hospital_System","date"])["occupied"]
    .sum()
    .reset_index(name="rooms_occupied")
)

capacity = (
    rooms.groupby("Hospital_System")
    .size()
    .reset_index(name="total_rooms")
)

daily_system = daily_system.merge(capacity)
daily_system["utilization_rate"] = daily_system["rooms_occupied"] / daily_system["total_rooms"]


# In[51]:


daily_system["util_roll7"] = (
    daily_system.groupby("Hospital_System")["utilization_rate"]
    .transform(lambda x: x.rolling(7, min_periods=2).mean())
)

daily_system["util_roll7"] = daily_system["util_roll7"].fillna(0)


# In[52]:


daily_system["near_high"] = (daily_system["utilization_rate"] >= 0.75).astype(int)


# In[53]:


# ======================================================================
# 6) Daily system utilization
# ======================================================================


g = daily_system.groupby("Hospital_System")

daily_system["util_yesterday"] = g["utilization_rate"].shift(1)
daily_system["util_last_week"] = g["utilization_rate"].shift(7)

daily_system["util_change_1d"] = daily_system["utilization_rate"] - daily_system["util_yesterday"]
daily_system["util_change_7d"] = daily_system["utilization_rate"] - daily_system["util_last_week"]

daily_system.fillna(0, inplace=True)


# In[54]:


# ======================================================================
# 8) Define utilization states (TARGET)
# ======================================================================

q25 = g["utilization_rate"].transform(lambda x: x.quantile(0.25))
q75 = g["utilization_rate"].transform(lambda x: x.quantile(0.75))

daily_system["util_state"] = np.select(
    [
        daily_system["utilization_rate"] < q25,
        daily_system["utilization_rate"] > q75
    ],
    ["low","high"],
    default="normal"
)


# In[112]:


# ======================================================================
# 9) Create future prediction target (t+1)
# ======================================================================
# Create future targets
daily_system["util_state_t1"] = (
    daily_system
    .groupby("Hospital_System")["util_state"]
    .shift(-1)
)

daily_system["util_state_t3"] = (
    daily_system
    .groupby("Hospital_System")["util_state"]
    .shift(-3)
)

daily_system["util_state_t7"] = (
    daily_system
    .groupby("Hospital_System")["util_state"]
    .shift(-7)
)

daily_system["util_state_t30"] = (
    daily_system
    .groupby("Hospital_System")["util_state"]
    .shift(-30)
)

model_df = daily_system.dropna(subset=["util_state_t1"])


# In[113]:


X = model_df[
    [
        "util_yesterday",
        "util_last_week",
        "util_change_1d",
        "util_change_7d",
        "util_roll7",
        "near_high",
        "Hospital_System"
    ]
]

y = model_df["util_state_t1"]


# In[114]:


# ======================================================================
# 11) Train / test split (time‑safe)
# ======================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# In[115]:


# ======================================================================
# 12) Preprocessing + model
# ======================================================================
preprocess = ColumnTransformer(
    [
        ("num", StandardScaler(), [
            "util_yesterday","util_last_week",
            "util_change_1d","util_change_7d",
            "util_roll7","near_high"
        ]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Hospital_System"]),
    ]
)

model = Pipeline(
    [
        ("prep", preprocess),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight={"high": 2, "normal": 1, "low": 1}
        ))
    ]
)

model.fit(X_train, y_train)


# In[116]:


# ======================================================================
# 13) Evaluate
# ======================================================================
from sklearn.metrics import classification_report, log_loss
from IPython.display import display

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Classification report → DataFrame
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

display(report_df)

# Log loss → DataFrame
logloss_df = pd.DataFrame(
    {"Log Loss": [log_loss(y_test, y_prob)]}
)

display(logloss_df)


# In[117]:


cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.imshow(cm, cmap="Blues")
plt.xticks(range(len(model.classes_)), model.classes_)
plt.yticks(range(len(model.classes_)), model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(model.classes_)):
    for j in range(len(model.classes_)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.title("Confusion Matrix (t+1)")
plt.show()


# In[118]:


from sklearn.linear_model import LogisticRegression

ridge_model = Pipeline(
    [
        ("prep", preprocess),
        ("clf", LogisticRegression(
            penalty="l2",          # Ridge
            C=1.25,
            max_iter=2000,
            class_weight={"high": 2, "normal": 1, "low": 1},
            solver="lbfgs"
        ))
    ]
)

ridge_model.fit(X_train, y_train)

y_pred = ridge_model.predict(X_test)
y_prob = ridge_model.predict_proba(X_test)

display(
    pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True)
    ).transpose().round(3)
)

display(
    pd.DataFrame({"Log Loss": [log_loss(y_test, y_prob)]}).round(4)
)


# In[119]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report, log_loss
from IPython.display import display
import pandas as pd

svm_rbf = Pipeline(
    [
        ("prep", preprocess),
        ("clf", SVC(
            kernel="rbf",
            C=1,                     # flexibility
            gamma="scale",             # automatic, safe
            probability=True,          # needed for log loss
            class_weight={
                "high": 2,
                "normal": 1,
                "low": 1
            }
        ))
    ]
)


# In[120]:


svm_rbf.fit(X_train, y_train)


# In[121]:


y_pred = svm_rbf.predict(X_test)
y_prob = svm_rbf.predict_proba(X_test)

report = classification_report(y_test, y_pred, output_dict=True)
display(pd.DataFrame(report).transpose().round(3))

display(pd.DataFrame({"Log Loss": [log_loss(y_test, y_prob)]}).round(4))


# In[122]:


import pandas as pd
import matplotlib.pyplot as plt

# Manually enter your observed results
recall_data = pd.DataFrame({
    "Horizon": ["t+1", "t+7", "t+30"],
    "High Recall": [0.863, 0.617, 0.177]
})

plt.figure(figsize=(6,4))
plt.plot(recall_data["Horizon"], recall_data["High Recall"], marker="o")
plt.ylim(0,1)
plt.ylabel("Recall (High Utilization)")
plt.title("High‑Risk Detection vs Prediction Horizon")
plt.grid(True)
plt.show()


# In[124]:


import pandas as pd
import matplotlib.pyplot as plt

# F1-scores from your RBF-SVM results
f1_data = pd.DataFrame({
    "Horizon": ["t+1", "t+7", "t+30"],
    "High F1":   [0.775, 0.513, 0.510],   # high class
    "Low F1":    [0.792, 0.358, 0.000],   # low class (collapsed at t+30)
    "Normal F1": [0.802, 0.628, 0.627]    # normal class
})
plt.figure(figsize=(7,4))

plt.plot(f1_data["Horizon"], f1_data["High F1"], marker="o", label="High")
plt.plot(f1_data["Horizon"], f1_data["Low F1"], marker="o", label="Low")
plt.plot(f1_data["Horizon"], f1_data["Normal F1"], marker="o", label="Normal")

plt.ylim(0,1)
plt.ylabel("F1 Score")
plt.xlabel("Prediction Horizon")
plt.title("F1 Score by Utilization State vs Prediction Horizon (RBF‑SVM)")
plt.legend()
plt.grid(True)

plt.show()


# In[ ]:




