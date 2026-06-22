import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import joblib

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# -------------------------------------------------
# CREATE OUTPUT FOLDERS
# -------------------------------------------------
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -------------------------------------------------
# LOAD CSV FILES
# -------------------------------------------------
features_df = pd.read_csv("outputs/vascular_features.csv")
pred_df = pd.read_csv("outputs/predictions.csv")

print("✅ Files loaded")

# -------------------------------------------------
# CLEAN IMAGE NAMES
# -------------------------------------------------
features_df["base"] = features_df["image"].str.replace(
    "_vessel.png",
    "",
    regex=False
)

pred_df["base"] = pred_df["image_path"].apply(
    lambda x: x.split("/")[-1].split("\\")[-1].split(".")[0]
)

# -------------------------------------------------
# MERGE DATA
# -------------------------------------------------
df = pd.merge(features_df, pred_df, on="base")

print("✅ Merged samples:", len(df))

# -------------------------------------------------
# CHECK COLUMN NAMES
# -------------------------------------------------
print("\n📌 Available Columns:")
print(df.columns)

# -------------------------------------------------
# FEATURE COLUMNS
# -------------------------------------------------
feature_cols = [
    "vessel_density",
    "vessel_length",
    "branching_points",
    "fractal_dimension",
    "tortuosity",
    "avg_thickness"
]

X = df[feature_cols]

# -------------------------------------------------
# TARGET COLUMN
# -------------------------------------------------
# CHANGE "age" BELOW IF YOUR CSV
# USES A DIFFERENT COLUMN NAME
# -------------------------------------------------
y = df["true_age"]

# -------------------------------------------------
# TRAIN TEST SPLIT
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("✅ Train-test split done")

# -------------------------------------------------
# XGBOOST MODEL
# -------------------------------------------------
model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

print("✅ XGBoost model trained")

# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------
model_path = "model/models/xgboost_biomarker_model.pkl"

joblib.dump(model, model_path)

print(f"✅ Saved model: {model_path}")

# -------------------------------------------------
# EVALUATION
# -------------------------------------------------
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)

print(f"\n✅ Biomarker Model MAE: {mae:.4f}")

# -------------------------------------------------
# SHAP EXPLAINER
# -------------------------------------------------
explainer = shap.Explainer(model)

shap_values = explainer(X_test)

print("✅ SHAP values computed")

# -------------------------------------------------
# GLOBAL SHAP SUMMARY PLOT
# -------------------------------------------------
plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.tight_layout()

summary_path = "outputs/shap_summary.png"

plt.savefig(
    summary_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"✅ Saved: {summary_path}")

# -------------------------------------------------
# GLOBAL SHAP BAR PLOT
# -------------------------------------------------
plt.figure()

shap.plots.bar(
    shap_values,
    show=False
)

plt.tight_layout()

bar_path = "outputs/shap_bar.png"

plt.savefig(
    bar_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"✅ Saved: {bar_path}")

# -------------------------------------------------
# LOCAL EXPLANATION
# -------------------------------------------------
# -------------------------------------------------
# LOCAL EXPLANATION
# -------------------------------------------------

sample_index = 289

sample = X_test.iloc[sample_index]

sample_shap = shap_values.values[sample_index]

# -------------------------------------------------
# TRUE + PREDICTED AGE
# -------------------------------------------------

true_age = y_test.iloc[sample_index]

predicted_age = model.predict(
    sample.values.reshape(1, -1)
)[0]

rag = predicted_age - true_age

print(f"\n✅ Actual Age: {true_age:.1f}")
print(f"✅ Predicted Age: {predicted_age:.1f}")
print(f"✅ Retinal Age Gap: {rag:.1f}")

# -------------------------------------------------
# BIOMARKER STATUS FUNCTION
# -------------------------------------------------

def get_status(feature_name, value):

    mean = df[feature_name].mean()

    std = df[feature_name].std()

    if value > mean + std:
        return "High"

    elif value < mean - std:
        return "Low"

    else:
        return "Normal"

# -------------------------------------------------
# CREATE BIOMARKER TABLE
# -------------------------------------------------

biomarker_rows = []

for i, feat in enumerate(feature_cols):

    value = sample[feat]

    shap_val = sample_shap[i]

    status = get_status(feat, value)

    contribution = (
        "Increased Age"
        if shap_val > 0
        else "Decreased Age"
    )

    biomarker_rows.append([
        feat,
        round(value, 4),
        status,
        round(shap_val, 2),
        contribution
    ])

biomarker_df = pd.DataFrame(
    biomarker_rows,
    columns=[
        "Biomarker",
        "Value",
        "Status",
        "SHAP Contribution",
        "Effect"
    ]
)

print("\n📊 BIOMARKER ANALYSIS\n")

print(biomarker_df)

# -------------------------------------------------
# SAVE TABLE
# -------------------------------------------------

biomarker_df.to_csv(
    "outputs/biomarker_analysis.csv",
    index=False
)

print("\n✅ Saved biomarker table")

# -------------------------------------------------
# SORT FEATURES
# -------------------------------------------------

sorted_idx = np.argsort(
    np.abs(sample_shap)
)[::-1]

sorted_features = np.array(
    feature_cols
)[sorted_idx]

sorted_values = sample_shap[sorted_idx]

# -------------------------------------------------
# LOCAL BAR GRAPH
# -------------------------------------------------

plt.figure(figsize=(10, 6))

colors = [
    "red" if v > 0 else "blue"
    for v in sorted_values
]

plt.barh(
    sorted_features,
    sorted_values,
    color=colors
)

plt.xlabel("Contribution to Predicted Age")

plt.title(
    f"Local Biomarker Explanation\n"
    f"Actual Age: {true_age:.1f} | "
    f"Predicted Age: {predicted_age:.1f} | "
    f"RAG: {rag:.1f}"
)

plt.gca().invert_yaxis()

plt.axvline(0, color="black")

plt.tight_layout()

local_path = (
    "outputs/local_feature_explanation.png"
)

plt.savefig(
    local_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"\n✅ Saved: {local_path}")

# -------------------------------------------------
# AI-STYLE MEDICAL INTERPRETATION
# -------------------------------------------------

report = []

if rag > 5:
    report.append(
        "Retinal age appears significantly elevated "
        "relative to chronological age."
    )

elif rag > 2:
    report.append(
        "Mild elevation in retinal aging detected."
    )

else:
    report.append(
        "Retinal aging appears within normal range."
    )

# -------------------------------------------------

for _, row in biomarker_df.iterrows():

    if row["Effect"] == "Increased Age":

        report.append(
            f"{row['Biomarker']} "
            f"({row['Status']}) contributed "
            f"to increased retinal age prediction."
        )

# -------------------------------------------------

final_report = "\n".join(report)

print("\n🩺 MEDICAL INTERPRETATION\n")

print(final_report)

# -------------------------------------------------
# SAVE REPORT
# -------------------------------------------------

with open(
    "outputs/medical_report.txt",
    "w"
) as f:

    f.write(final_report)

print("\n✅ Saved medical report")
