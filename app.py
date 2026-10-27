import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

print("AI-Powered Customer Churn Prediction & Customer Analytics Platform")

print("Current Folder:", os.getcwd())
print("Files:", os.listdir())

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())
print(df.shape)
# 3. Remove extra spaces from column names
df.columns = df.columns.str.strip()

# 4. Check missing values before cleaning
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# 5. Convert TotalCharges into numeric format
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce"
)

# 6. Check and remove duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()

# 7. Remove rows with missing values
df = df.dropna()

# 8. Check remaining missing values
print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sum())

# 9. Check final dataset
print("\nCleaned Dataset Shape:", df.shape)
print("\nChurn Distribution:")
print(df["Churn"].value_counts())

# 10. Save cleaned dataset
df.to_csv("telco_churn_cleaned.csv", index=False)

print("\nData cleaning completed successfully!")
print("Cleaned dataset saved as: telco_churn_cleaned.csv")




#STEP 2: EXPLORATORY DATA ANALYSIS
# ==============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("telco_churn_cleaned.csv")

print("\n--- EDA STARTED ---")
print("Dataset Shape:", df.shape)

print("\nChurn Counts:")
print(df["Churn"].value_counts())

# Set chart style
sns.set_theme(style="whitegrid")

#Graph 1: Customer Churn Distribution

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (Yes = Left, No = Stayed)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("churn_distribution.png")
plt.show()



#Step 3 — Graph 2: Contract Type vs Churn

plt.figure(figsize=(9, 5))

sns.countplot(data=df, x="Contract", hue="Churn")

plt.title("Contract Type vs Customer Churn")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig("contract_vs_churn.png")
plt.show()
#Step 4 — Graph 3: Tenure vs Churn

plt.figure(figsize=(7, 5))

sns.boxplot(data=df, x="Churn", y="tenure")

plt.title("Customer Tenure vs Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")

plt.tight_layout()
plt.savefig("tenure_vs_churn.png")
plt.show()


#Step 5 — Churn rate calculate karo

churn_rate = (df["Churn"] == "Yes").mean() * 100

print(f"Customer Churn Rate: {churn_rate:.2f}%")



# STEP 3: FEATURE ENGINEERING & MODEL TRAINING
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("\n--- MACHINE LEARNING STARTED ---")

# Load cleaned dataset
df = pd.read_csv("telco_churn_cleaned.csv")

# Remove customer ID because it is not useful for prediction
df = df.drop(columns=["customerID"], errors="ignore")

# Convert target column into numbers
# No = 0, Yes = 1
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Separate input features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

# Prepare preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_cols),
        ("numerical", StandardScaler(), numerical_cols)
    ]
)

# Create machine learning pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\n--- MODEL RESULTS ---")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Stayed", "Churned"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nModel training completed successfully!")






# ==========================================
# STEP 4: MODEL COMPARISON & MODEL SAVING
# ==========================================

import joblib
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

print("\n--- MODEL COMPARISON STARTED ---")

# Use the same preprocessor and train/test split from Step 3
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=12,
        random_state=42, n_jobs=-1
    )
}

results = []

for name, classifier in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "F1 Score": round(f1, 4)
    })

    print(f"\n{name}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score: {f1:.4f}")

# Display comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1 Score", ascending=False
)

print("\n--- MODEL COMPARISON RESULTS ---")
print(results_df.to_string(index=False))

# Select the model with the highest F1 score
best_model_name = results_df.iloc[0]["Model"]
best_classifier = models[best_model_name]

best_model = Pipeline(steps=[
    ("preprocessor", clone(preprocessor)),
    ("classifier", best_classifier)
])

best_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(best_model, "best_churn_model.pkl")

# Save model comparison results
results_df.to_csv("model_comparison.csv", index=False)

print(f"\nBest Model: {best_model_name}")
print("Saved model: best_churn_model.pkl")
print("Saved comparison: model_comparison.csv")
