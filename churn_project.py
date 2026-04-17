import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# LOAD AND INSPECT THE DATA

df = pd.read_csv("/Users/vigneshkhajuria/Downloads/Churn_Modelling.csv")

print("Shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass balance (Exited):")
print(df["Exited"].value_counts())
print(df["Exited"].value_counts(normalize=True).round(3))

# CLEAN THE DATA


df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

print("Columns after cleaning:")
print(df.columns.tolist())

print("\nShape after cleaning:", df.shape)

# EDA


fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Churn Patterns by Customer Segment", fontsize=16,
             fontweight="bold")

# Chart 1: Churn by Geography
geo_churn = df.groupby("Geography")["Exited"].mean().reset_index()
axes[0, 0].bar(geo_churn["Geography"], geo_churn["Exited"],
               color=["#4C72B0", "#DD8452", "#55A868"])
axes[0, 0].set_title("Churn Rate by Geography")
axes[0, 0].set_ylabel("Churn Rate")
axes[0, 0].set_ylim(0, 0.5)
for i, v in enumerate(geo_churn["Exited"]):
    axes[0, 0].text(i, v + 0.01, f"{v:.1%}", ha="center", fontweight="bold")

# Chart 2: Churn by Gender
gen_churn = df.groupby("Gender")["Exited"].mean().reset_index()
axes[0, 1].bar(gen_churn["Gender"], gen_churn["Exited"],
               color=["#4C72B0", "#DD8452"])
axes[0, 1].set_title("Churn Rate by Gender")
axes[0, 1].set_ylabel("Churn Rate")
axes[0, 1].set_ylim(0, 0.5)
for i, v in enumerate(gen_churn["Exited"]):
    axes[0, 1].text(i, v + 0.01, f"{v:.1%}", ha="center", fontweight="bold")

# Chart 3: Churn by Active Member Status
act_churn = df.groupby("IsActiveMember")["Exited"].mean().reset_index()
act_churn["IsActiveMember"] = act_churn["IsActiveMember"].map(
    {0: "Inactive", 1: "Active"})
axes[0, 2].bar(act_churn["IsActiveMember"], act_churn["Exited"],
               color=["#DD8452", "#55A868"])
axes[0, 2].set_title("Churn Rate: Active vs Inactive")
axes[0, 2].set_ylabel("Churn Rate")
axes[0, 2].set_ylim(0, 0.5)
for i, v in enumerate(act_churn["Exited"]):
    axes[0, 2].text(i, v + 0.01, f"{v:.1%}", ha="center", fontweight="bold")

# Chart 4: Churn by Number of Products
prod_churn = df.groupby("NumOfProducts")["Exited"].mean().reset_index()
axes[1, 0].bar(prod_churn["NumOfProducts"].astype(str), prod_churn["Exited"],
               color="#4C72B0")
axes[1, 0].set_title("Churn Rate by Number of Products")
axes[1, 0].set_ylabel("Churn Rate")
axes[1, 0].set_xlabel("Number of Products")
axes[1, 0].set_ylim(0, 1.1)
for i, v in enumerate(prod_churn["Exited"]):
    axes[1, 0].text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")

# Chart 5: Age distribution by churn
df[df["Exited"] == 0]["Age"].plot(kind="hist", alpha=0.6, label="Retained",
                                  ax=axes[1, 1], bins=30, color="#55A868")
df[df["Exited"] == 1]["Age"].plot(kind="hist", alpha=0.6, label="Churned",
                                  ax=axes[1, 1], bins=30, color="#DD8452")
axes[1, 1].set_title("Age Distribution: Churned vs Retained")
axes[1, 1].set_xlabel("Age")
axes[1, 1].set_ylabel("Count")
axes[1, 1].legend()

# Chart 6: Balance distribution by churn
df[df["Exited"] == 0]["Balance"].plot(kind="hist", alpha=0.6, label="Retained",
                                      ax=axes[1, 2], bins=30, color="#55A868")
df[df["Exited"] == 1]["Balance"].plot(kind="hist", alpha=0.6, label="Churned",
                                      ax=axes[1, 2], bins=30, color="#DD8452")
axes[1, 2].set_title("Balance Distribution: Churned vs Retained")
axes[1, 2].set_xlabel("Balance")
axes[1, 2].set_ylabel("Count")
axes[1, 2].legend()

plt.tight_layout()
plt.savefig("eda_charts.png", dpi=150, bbox_inches="tight")
plt.show()
print("EDA charts saved as eda_charts.png")

# Print key EDA findings as numbers
print("\n--- Key EDA Findings ---")
print("\nChurn rate by Geography:")
print(df.groupby("Geography")["Exited"].mean().round(3))

print("\nChurn rate by Gender:")
print(df.groupby("Gender")["Exited"].mean().round(3))

print("\nChurn rate by Active Status:")
print(df.groupby("IsActiveMember")["Exited"].mean().round(3))

print("\nChurn rate by Num of Products:")
print(df.groupby("NumOfProducts")["Exited"].mean().round(3))

print("\nAverage age — Churned vs Retained:")
print(df.groupby("Exited")["Age"].mean().round(1))


# STEP 4: FEATURE ENGINEERING

# Age Band
def age_band(age):
    if age < 30:
        return "Under 30"
    elif age < 45:
        return "30-44"
    elif age < 60:
        return "45-59"
    else:
        return "60+"


df["AgeBand"] = df["Age"].apply(age_band)


# Balance Segment
def balance_segment(bal):
    if bal == 0:
        return "Zero"
    elif bal < 50000:
        return "Low"
    elif bal < 125000:
        return "Medium"
    else:
        return "High"


df["BalanceSegment"] = df["Balance"].apply(balance_segment)

# Inactive Flag
# 1 means inactive — we saw this group churns at nearly 2x the rate
df["InactiveFlag"] = (df["IsActiveMember"] == 0).astype(int)

# Multi Product Flag
df["MultiProductFlag"] = (df["NumOfProducts"] >= 3).astype(int)

# Verify the new features
print("New features added:")
print(df[["Age", "AgeBand", "Balance", "BalanceSegment", "InactiveFlag",
          "MultiProductFlag"]].head(10))

print("\nChurn rate by Age Band:")
print(df.groupby("AgeBand")["Exited"].mean().round(3))

print("\nChurn rate by Balance Segment:")
print(df.groupby("BalanceSegment")["Exited"].mean().round(3))

print("\nMultiProductFlag distribution:")
print(df["MultiProductFlag"].value_counts())

print("\nChurn rate — MultiProductFlag:")
print(df.groupby("MultiProductFlag")["Exited"].mean().round(3))

# PREPARE FEATURES, ENCODE, SPLIT

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

X = df.drop(columns=["Exited"])
y = df["Exited"]

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

categorical_cols = ["Geography", "Gender", "AgeBand", "BalanceSegment"]
numerical_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                  "HasCrCard", "IsActiveMember", "EstimatedSalary",
                  "InactiveFlag", "MultiProductFlag"]

print("\nCategorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     categorical_cols),
    ("num", StandardScaler(), numerical_cols)
])

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain set size:", X_train.shape)
print("Test set size: ", X_test.shape)

print("\nChurn rate in train set:", y_train.mean().round(3))
print("Churn rate in test set: ", y_test.mean().round(3))

# TRAIN BOTH MODELS

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Model 1: Logistic Regression
lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000, random_state=42))
])

lr_pipeline.fit(X_train, y_train)
print("Logistic Regression — training complete")

# Model 2: Random Forest
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

rf_pipeline.fit(X_train, y_train)
print("Random Forest — training complete")

print("\nBoth models trained successfully.")
print("Next step: evaluate them on the test set.")

# EVALUATE BOTH MODELS

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)


def evaluate_model(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'=' * 45}")
    print(f"  {name}")
    print(f"{'=' * 45}")
    print(f"  Accuracy  : {accuracy:.4f}  — overall correct predictions")
    print(
        f"  Precision : {precision:.4f}  — of predicted churners, how many actually churned")
    print(
        f"  Recall    : {recall:.4f}  — of actual churners, how many did we catch")
    print(f"  F1 Score  : {f1:.4f}  — balance between precision and recall")
    print(
        f"  ROC-AUC   : {roc_auc:.4f}  — overall ranking ability (higher = better)")

    return y_pred, y_prob, roc_auc


# Run evaluation for both models
lr_pred, lr_prob, lr_auc = evaluate_model("Logistic Regression", lr_pipeline,
                                          X_test, y_test)
rf_pred, rf_prob, rf_auc = evaluate_model("Random Forest", rf_pipeline, X_test,
                                          y_test)

# Confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

ConfusionMatrixDisplay(confusion_matrix(y_test, lr_pred),
                       display_labels=["Retained", "Churned"]).plot(ax=axes[0],
                                                                    colorbar=False)
axes[0].set_title("Logistic Regression")

ConfusionMatrixDisplay(confusion_matrix(y_test, rf_pred),
                       display_labels=["Retained", "Churned"]).plot(ax=axes[1],
                                                                    colorbar=False)
axes[1].set_title("Random Forest")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# ROC Curve comparison
from sklearn.metrics import roc_curve

fig, ax = plt.subplots(figsize=(8, 6))

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

ax.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {lr_auc:.3f})",
        color="#4C72B0")
ax.plot(fpr_rf, tpr_rf, label=f"Random Forest       (AUC = {rf_auc:.3f})",
        color="#DD8452")
ax.plot([0, 1], [0, 1], "k--", label="Random baseline")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# Model selection
print("\n" + "=" * 45)
print("  MODEL COMPARISON SUMMARY")
print("=" * 45)
print(f"  Logistic Regression ROC-AUC : {lr_auc:.4f}")
print(f"  Random Forest ROC-AUC       : {rf_auc:.4f}")

if rf_auc >= lr_auc:
    print("\n  Winner: Random Forest")
    print(
        "  Reason: Higher ROC-AUC — better at ranking customers by churn risk")
    best_pipeline = rf_pipeline
    best_pred = rf_pred
    best_prob = rf_prob
    best_name = "Random Forest"
else:
    print("\n  Winner: Logistic Regression")
    print("  Reason: Higher ROC-AUC — stronger overall discrimination")
    best_pipeline = lr_pipeline
    best_pred = lr_pred
    best_prob = lr_prob
    best_name = "Logistic Regression"

print(f"\n  '{best_name}' will be used for scoring.")

# SCORE ALL CUSTOMERS + EXPORT CSVs

full_prob = best_pipeline.predict_proba(X)[:, 1]
full_pred = best_pipeline.predict(X)


# --- Assign Risk Bands ---
def risk_band(prob):
    if prob >= 0.70:
        return "High Risk"
    elif prob >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"


risk_labels = [risk_band(p) for p in full_prob]

# Build the scored dataframe
scored_df = df.copy()
scored_df["ActualChurn"] = y.values
scored_df["PredictedChurnLabel"] = full_pred
scored_df["PredictedChurnProb"] = full_prob.round(4)
scored_df["RiskBand"] = risk_labels

print("Scored dataset shape:", scored_df.shape)
print("\nRisk Band distribution:")
print(scored_df["RiskBand"].value_counts())

print("\nSample of scored output:")
print(scored_df[["Age", "Geography", "Balance", "ActualChurn",
                 "PredictedChurnProb", "RiskBand"]].head(10))

# Full scored CSV for Tableau
scored_df.to_csv("scored_bank_churn.csv", index=False)
print("\nExported: scored_bank_churn.csv")

# Export 2: Feature importances from Random Forest
cat_feature_names = (
    best_pipeline.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .get_feature_names_out(categorical_cols)
    .tolist()
)
all_feature_names = cat_feature_names + numerical_cols

importances = best_pipeline.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False).reset_index(drop=True)

importance_df["Importance"] = importance_df["Importance"].round(4)

print("\nTop 15 Feature Importances:")
print(importance_df.head(15).to_string(index=False))

importance_df.to_csv("model_feature_importance.csv", index=False)
print("\nExported: model_feature_importance.csv")

# BUSINESS SUMMARY

print("\n" + "=" * 50)
print("  BUSINESS SUMMARY")
print("=" * 50)
print(f"""
Model used       : Random Forest (ROC-AUC: {rf_auc:.3f})

Key churn drivers identified:
  - Age is the strongest predictor: customers aged
    45-59 churn at 49.4%, vs 7.6% for Under 30s
  - Germany has double the churn rate of France/Spain
    (32.4% vs ~16%)
  - Customers with 3+ products churn at 85.9% —
    over-selling is a major retention risk
  - Inactive members churn at 26.9% vs 14.3% for
    active members — engagement is a strong signal
  - Female customers churn at 25.1% vs 16.5% for males

Risk band breakdown:
""")
print(scored_df["RiskBand"].value_counts().to_string())
print(f"""
Recommendation:
  Prioritise the High Risk segment for immediate
  retention outreach. Focus especially on inactive
  customers aged 45-59 in Germany with 1 product.
""")
