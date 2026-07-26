"""
=======================================================
  HR Employee Attrition Prediction System
  Author: Your Name | Internship Project
=======================================================

Steps:
  1. Data Loading & Exploration (EDA)
  2. Preprocessing & Feature Engineering
  3. Model Training (Logistic Regression, Random Forest, XGBoost)
  4. Evaluation (Accuracy, F1, ROC-AUC, Confusion Matrix)
  5. Feature Importance
  6. Prediction on new employee data
=======================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
import joblib
import os

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data(filepath=None):
    """Load dataset from file or generate IBM HR sample data."""
    if filepath and os.path.exists(filepath):
        print(f"[INFO] Loading dataset from: {filepath}")
        df = pd.read_csv(filepath)
    else:
        print("[INFO] No dataset file found. Generating IBM HR Analytics sample data...")
        df = generate_ibm_hr_data()
    return df


def generate_ibm_hr_data(n=1470, seed=42):
    """Generate a realistic IBM HR-style dataset."""
    np.random.seed(seed)
    n_leave = int(n * 0.16)    # ~16% attrition rate
    n_stay  = n - n_leave

    def make_group(size, leaving):
        age         = np.random.randint(22, 35, size) if leaving else np.random.randint(28, 58, size)
        monthly_inc = np.random.randint(1500, 5000, size) if leaving else np.random.randint(3000, 20000, size)
        wlb         = np.random.choice([1, 2], size=size) if leaving else np.random.choice([3, 4], size=size)
        overtime    = np.random.choice(["Yes", "No"], size=size, p=[0.6, 0.4] if leaving else [0.2, 0.8])
        job_sat     = np.random.choice([1, 2], size=size) if leaving else np.random.choice([3, 4], size=size)
        return pd.DataFrame({
            "Age":                    age,
            "Attrition":              ["Yes"] * size if leaving else ["No"] * size,
            "BusinessTravel":         np.random.choice(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], size),
            "DailyRate":              np.random.randint(100, 1500, size),
            "Department":             np.random.choice(["Sales", "Research & Development", "Human Resources"], size),
            "DistanceFromHome":       np.random.randint(1, 30, size),
            "Education":              np.random.randint(1, 6, size),
            "EducationField":         np.random.choice(["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"], size),
            "EmployeeCount":          1,
            "EmployeeNumber":         np.random.randint(1, 2000, size),
            "EnvironmentSatisfaction": np.random.randint(1, 5, size),
            "Gender":                 np.random.choice(["Male", "Female"], size),
            "HourlyRate":             np.random.randint(30, 100, size),
            "JobInvolvement":         np.random.randint(1, 5, size),
            "JobLevel":               np.random.choice([1, 2] if leaving else [2, 3, 4, 5], size),
            "JobRole":                np.random.choice(["Sales Executive", "Research Scientist", "Lab Technician",
                                                         "Manufacturing Director", "Healthcare Rep", "Manager"], size),
            "JobSatisfaction":        job_sat,
            "MaritalStatus":          np.random.choice(["Single", "Married", "Divorced"], size),
            "MonthlyIncome":          monthly_inc,
            "MonthlyRate":            np.random.randint(2000, 27000, size),
            "NumCompaniesWorked":     np.random.randint(0, 10, size),
            "Over18":                 "Y",
            "OverTime":               overtime,
            "PercentSalaryHike":      np.random.randint(11, 25, size),
            "PerformanceRating":      np.random.choice([3, 4], size),
            "RelationshipSatisfaction": np.random.randint(1, 5, size),
            "StandardHours":          80,
            "StockOptionLevel":       np.random.randint(0, 4, size),
            "TotalWorkingYears":      np.random.randint(0, 5, size) if leaving else np.random.randint(3, 35, size),
            "TrainingTimesLastYear":  np.random.randint(0, 7, size),
            "WorkLifeBalance":        wlb,
            "YearsAtCompany":         np.random.randint(0, 3, size) if leaving else np.random.randint(1, 30, size),
            "YearsInCurrentRole":     np.random.randint(0, 3, size) if leaving else np.random.randint(1, 15, size),
            "YearsSinceLastPromotion": np.random.randint(0, 10, size),
            "YearsWithCurrManager":   np.random.randint(0, 3, size) if leaving else np.random.randint(1, 15, size),
        })

    df = pd.concat([make_group(n_leave, True), make_group(n_stay, False)], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────

def eda(df, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "="*60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("="*60)
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nAttrition distribution:\n{df['Attrition'].value_counts()}")
    print(f"\nAttrition rate: {df['Attrition'].value_counts(normalize=True)['Yes']*100:.1f}%")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("HR Employee Attrition — Exploratory Analysis", fontsize=16, fontweight="bold")

    # 1. Attrition distribution
    counts = df["Attrition"].value_counts()
    axes[0, 0].bar(counts.index, counts.values,
                   color=["#e74c3c", "#2ecc71"], edgecolor="white", linewidth=1.5)
    axes[0, 0].set_title("Attrition Distribution")
    axes[0, 0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0, 0].text(i, v + 5, str(v), ha="center", fontweight="bold")

    # 2. Age by Attrition
    df.boxplot(column="Age", by="Attrition", ax=axes[0, 1])
    axes[0, 1].set_title("Age by Attrition")
    axes[0, 1].set_xlabel("")
    plt.sca(axes[0, 1])
    plt.title("Age by Attrition")

    # 3. MonthlyIncome by Attrition
    df.boxplot(column="MonthlyIncome", by="Attrition", ax=axes[0, 2])
    axes[0, 2].set_title("Monthly Income by Attrition")
    axes[0, 2].set_xlabel("")
    plt.sca(axes[0, 2])
    plt.title("Monthly Income by Attrition")

    # 4. OverTime vs Attrition
    ot = df.groupby(["OverTime", "Attrition"]).size().unstack()
    ot.plot(kind="bar", ax=axes[1, 0], color=["#2ecc71", "#e74c3c"], edgecolor="white")
    axes[1, 0].set_title("OverTime vs Attrition")
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)
    axes[1, 0].legend(["No", "Yes"])

    # 5. Department vs Attrition
    dept = df.groupby(["Department", "Attrition"]).size().unstack(fill_value=0)
    dept.plot(kind="bar", ax=axes[1, 1], color=["#2ecc71", "#e74c3c"], edgecolor="white")
    axes[1, 1].set_title("Department vs Attrition")
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=15)
    axes[1, 1].legend(["No", "Yes"])

    # 6. Job Satisfaction vs Attrition
    js = df.groupby(["JobSatisfaction", "Attrition"]).size().unstack(fill_value=0)
    js.plot(kind="bar", ax=axes[1, 2], color=["#2ecc71", "#e74c3c"], edgecolor="white")
    axes[1, 2].set_title("Job Satisfaction vs Attrition")
    axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), rotation=0)
    axes[1, 2].legend(["No", "Yes"])

    plt.tight_layout()
    fig.savefig(f"{output_dir}/eda_plots.png", dpi=150, bbox_inches="tight")
    print(f"\n[INFO] EDA plots saved → {output_dir}/eda_plots.png")
    plt.close()


# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df):
    print("\n" + "="*60)
    print("  PREPROCESSING & FEATURE ENGINEERING")
    print("="*60)

    df = df.copy()

    # Drop constant/ID columns
    drop_cols = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Encode target
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

    # Encode binary categoricals
    df["OverTime"]  = (df["OverTime"] == "Yes").astype(int)
    df["Gender"]    = (df["Gender"] == "Male").astype(int)

    # One-Hot encode remaining categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    print(f"Final feature count: {df.shape[1] - 1}")
    print(f"Attrition class balance: {df['Attrition'].value_counts().to_dict()}")

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]
    return X, y, df


# ─────────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────────

def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
    }

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    trained = {}
    print("\n" + "="*60)
    print("  MODEL TRAINING (5-Fold Cross-Validation)")
    print("="*60)
    for name, model in models.items():
        X_in = X_train_sc if name == "Logistic Regression" else X_train
        cv_scores = cross_val_score(model, X_in, y_train, cv=5, scoring="roc_auc")
        model.fit(X_in, y_train)
        trained[name] = model
        print(f"  {name:<25} ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return trained, scaler


# ─────────────────────────────────────────────
# 5. EVALUATE MODELS
# ─────────────────────────────────────────────

def evaluate_models(trained, scaler, X_test, y_test, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("  MODEL EVALUATION ON TEST SET")
    print("="*60)

    results = {}
    X_test_sc = scaler.transform(X_test)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Model Evaluation — HR Attrition", fontsize=16, fontweight="bold")
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for idx, (name, model) in enumerate(trained.items()):
        X_in = X_test_sc if name == "Logistic Regression" else X_test
        y_pred = model.predict(X_in)
        y_prob = model.predict_proba(X_in)[:, 1]

        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred)
        auc    = roc_auc_score(y_test, y_prob)
        results[name] = {"Accuracy": acc, "F1-Score": f1, "ROC-AUC": auc}

        print(f"\n  ── {name} ──")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Stay", "Leave"]))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Stay", "Leave"])
        disp.plot(ax=axes[0, idx], colorbar=False, cmap="Blues")
        axes[0, idx].set_title(f"{name}\nAcc={acc:.3f}  F1={f1:.3f}")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[1, idx].plot(fpr, tpr, color=colors[idx], lw=2, label=f"AUC={auc:.3f}")
        axes[1, idx].plot([0,1],[0,1],"k--", alpha=0.4)
        axes[1, idx].set_xlabel("False Positive Rate")
        axes[1, idx].set_ylabel("True Positive Rate")
        axes[1, idx].set_title(f"ROC Curve — {name}")
        axes[1, idx].legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(f"{output_dir}/model_evaluation.png", dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Evaluation plots saved → {output_dir}/model_evaluation.png")
    plt.close()
    return results


# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def plot_feature_importance(trained, X_train, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    rf_model = trained["Random Forest"]

    importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
    top20 = importances.nlargest(20).sort_values()

    fig, ax = plt.subplots(figsize=(10, 8))
    top20.plot(kind="barh", ax=ax, color="#3498db", edgecolor="white")
    ax.set_title("Top 20 Feature Importances (Random Forest)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    fig.savefig(f"{output_dir}/feature_importance.png", dpi=150, bbox_inches="tight")
    print(f"[INFO] Feature importance plot saved → {output_dir}/feature_importance.png")
    plt.close()


# ─────────────────────────────────────────────
# 7. SUMMARY REPORT
# ─────────────────────────────────────────────

def print_summary(results):
    print("\n" + "="*60)
    print("  FINAL MODEL COMPARISON SUMMARY")
    print("="*60)
    summary = pd.DataFrame(results).T
    summary = summary.sort_values("ROC-AUC", ascending=False)
    print(summary.to_string(float_format="{:.4f}".format))
    best = summary.index[0]
    print(f"\n  ✅ Best Model: {best} (ROC-AUC = {summary.loc[best, 'ROC-AUC']:.4f})")
    return best


# ─────────────────────────────────────────────
# 8. PREDICT NEW EMPLOYEES
# ─────────────────────────────────────────────

def predict_new_employee(model, scaler, feature_columns, model_name):
    """
    Predict attrition risk for a single new employee.
    Provide the feature dict matching your training columns.
    """
    print("\n" + "="*60)
    print("  PREDICTING NEW EMPLOYEE ATTRITION RISK")
    print("="*60)

    # Example: a young employee with low salary, overtime, low satisfaction
    raw = {col: 0 for col in feature_columns}   # default all to 0
    # Override key fields (adjust to match your one-hot encoded column names)
    raw["Age"]               = 26
    raw["MonthlyIncome"]     = 2500
    raw["OverTime"]          = 1
    raw["JobSatisfaction"]   = 1
    raw["WorkLifeBalance"]   = 1
    raw["YearsAtCompany"]    = 1
    raw["TotalWorkingYears"] = 2
    raw["DistanceFromHome"]  = 20

    X_new = pd.DataFrame([raw])
    X_in  = scaler.transform(X_new) if model_name == "Logistic Regression" else X_new

    prob  = model.predict_proba(X_in)[0][1]
    pred  = "⚠️  HIGH RISK — Likely to Leave" if prob > 0.5 else "✅ LOW RISK — Likely to Stay"

    print(f"  Attrition Probability : {prob*100:.1f}%")
    print(f"  Prediction            : {pred}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR = "outputs"

    # 1. Load
    df = load_data(filepath="data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

    # 2. EDA
    eda(df, output_dir=OUTPUT_DIR)

    # 3. Preprocess
    X, y, processed_df = preprocess(df)

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[INFO] Train: {X_train.shape}  |  Test: {X_test.shape}")

    # 5. Train
    trained_models, scaler = train_models(X_train, y_train)

    # 6. Evaluate
    results = evaluate_models(trained_models, scaler, X_test, y_test, output_dir=OUTPUT_DIR)

    # 7. Feature Importance
    plot_feature_importance(trained_models, X_train, output_dir=OUTPUT_DIR)

    # 8. Summary
    best_name = print_summary(results)

    # 9. Save best model
    best_model = trained_models[best_name]
    joblib.dump(best_model, f"{OUTPUT_DIR}/best_model.pkl")
    joblib.dump(scaler,     f"{OUTPUT_DIR}/scaler.pkl")
    print(f"\n[INFO] Best model saved → {OUTPUT_DIR}/best_model.pkl")

    # 10. Predict a new employee
    predict_new_employee(best_model, scaler, X.columns.tolist(), best_name)

    print("\n" + "="*60)
    print("  ✅ PIPELINE COMPLETE — All outputs in /outputs/")
    print("="*60)
