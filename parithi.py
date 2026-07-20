# ==========================================================
# Asthma Disease Dataset Analysis
# ==========================================================

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

warnings.filterwarnings("ignore")

# ==========================================================
# Create Output Folder
# ==========================================================

output_folder = "Asthma Disease"
os.makedirs(output_folder, exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

dataset_path = r"C:\Users\DHARANISHWARAN\Downloads\asthma_dataset.csv"

df = pd.read_csv(dataset_path)

print("Dataset Loaded Successfully")
print(df.head())

# ==========================================================
# Dataset Information
# ==========================================================

with open(os.path.join(output_folder, "dataset_information.txt"), "w") as f:

    f.write("Dataset Shape\n")
    f.write(str(df.shape))

    f.write("\n\nColumns\n")
    f.write(str(df.columns.tolist()))

    f.write("\n\nData Types\n")
    f.write(str(df.dtypes))

# ==========================================================
# Missing Values
# ==========================================================

missing = df.isnull().sum()

missing.to_csv(
    os.path.join(output_folder, "missing_values.csv")
)

# ==========================================================
# Fill Missing Values
# ==========================================================

for col in df.columns:

    if pd.api.types.is_numeric_dtype(df[col]):

        df[col] = df[col].fillna(df[col].median())

    else:

        df[col] = df[col].fillna(df[col].mode()[0])

# ==========================================================
# Encode Categorical Columns
# ==========================================================

encoders = {}

for col in df.select_dtypes(include=["object"]).columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col].astype(str))

    encoders[col] = le

# ==========================================================
# Save Clean Dataset
# ==========================================================

df.to_csv(
    os.path.join(output_folder, "clean_dataset.csv"),
    index=False
)

# ==========================================================
# Summary Statistics
# ==========================================================

summary = df.describe(include="all")

summary.to_csv(
    os.path.join(output_folder, "summary_statistics.csv")
)

# ==========================================================
# Correlation Matrix
# ==========================================================

corr = df.corr(numeric_only=True)

corr.to_csv(
    os.path.join(output_folder, "correlation_matrix.csv")
)

plt.figure(figsize=(12,8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "correlation_heatmap.png")
)

plt.close()

# ==========================================================
# Histograms
# ==========================================================

numeric_df = df.select_dtypes(include=np.number)

for col in numeric_df.columns:

    plt.figure(figsize=(6,4))

    sns.histplot(
        numeric_df[col],
        kde=True,
        bins=20
    )

    plt.title(col)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            f"{col}_histogram.png"
        )
    )

    plt.close()

# ==========================================================
# Boxplots
# ==========================================================

for col in numeric_df.columns:

    plt.figure(figsize=(5,4))

    sns.boxplot(y=numeric_df[col])

    plt.title(col)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            f"{col}_boxplot.png"
        )
    )

    plt.close()

# ==========================================================
# Asthma Distribution
# ==========================================================

plt.figure(figsize=(6,6))

df["Asthma"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.ylabel("")

plt.title("Asthma Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "asthma_distribution.png"
    )
)

plt.close()

# ==========================================================
# Machine Learning
# ==========================================================

X = df.drop("Asthma", axis=1)

y = df["Asthma"]

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y

)

model = RandomForestClassifier(

    n_estimators=200,
    random_state=42

)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

report = classification_report(y_test, prediction)

with open(
    os.path.join(output_folder, "classification_report.txt"),
    "w"
) as f:

    f.write(f"Accuracy : {accuracy*100:.2f}%\n\n")
    f.write(report)

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_test, prediction)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "confusion_matrix.png"
    )
)

plt.close()

# ==========================================================
# ROC Curve
# ==========================================================

prob = model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.2f}"
)

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "ROC_Curve.png"
    )
)

plt.close()

# ==========================================================
# Feature Importance
# ==========================================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance.sort_values().plot(
    kind="barh",
    figsize=(8,6)
)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "feature_importance.png"
    )
)

plt.close()

# ==========================================================
# Save Model
# ==========================================================

joblib.dump(
    model,
    os.path.join(
        output_folder,
        "asthma_model.pkl"
    )
)

# ==========================================================
# Finished
# ==========================================================

print("\n======================================")
print("Analysis Completed Successfully")
print("Accuracy :", round(accuracy*100,2), "%")
print("Results saved in:")
print(os.path.abspath(output_folder))
print("======================================")
