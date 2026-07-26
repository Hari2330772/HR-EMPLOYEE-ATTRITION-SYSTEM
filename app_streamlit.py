"""
HR Employee Attrition — Interactive Dashboard
Run: streamlit run app_streamlit.py
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

if get_script_run_ctx() is None:
    print("This app must be launched with Streamlit.")
    print("Please run: streamlit run app_streamlit.py")
    raise SystemExit(0)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)

st.set_page_config(page_title="HR Attrition System", layout="wide", page_icon="👥")

# ── Sidebar ──────────────────────────────────
st.sidebar.title("⚙️ Configuration")
uploaded = st.sidebar.file_uploader("Upload CSV Dataset", type="csv")

# ── Helpers ──────────────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(42)
    n, n_leave = 1470, 235

    def make_group(size, leaving):
        return pd.DataFrame({
            "Age": np.random.randint(22, 35, size) if leaving else np.random.randint(28, 58, size),
            "Attrition": ["Yes"] * size if leaving else ["No"] * size,
            "BusinessTravel": np.random.choice(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], size),
            "Department": np.random.choice(["Sales", "Research & Development", "Human Resources"], size),
            "DistanceFromHome": np.random.randint(1, 30, size),
            "Education": np.random.randint(1, 6, size),
            "EducationField": np.random.choice(["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"], size),
            "EnvironmentSatisfaction": np.random.randint(1, 5, size),
            "Gender": np.random.choice(["Male", "Female"], size),
            "JobInvolvement": np.random.randint(1, 5, size),
            "JobLevel": np.random.choice([1, 2] if leaving else [2, 3, 4, 5], size),
            "JobRole": np.random.choice(["Sales Executive", "Research Scientist", "Lab Technician", "Manager"], size),
            "JobSatisfaction": np.random.choice([1, 2], size) if leaving else np.random.choice([3, 4], size),
            "MaritalStatus": np.random.choice(["Single", "Married", "Divorced"], size),
            "MonthlyIncome": np.random.randint(1500, 5000, size) if leaving else np.random.randint(3000, 20000, size),
            "NumCompaniesWorked": np.random.randint(0, 10, size),
            "OverTime": np.random.choice(["Yes", "No"], size, p=[0.6, 0.4] if leaving else [0.2, 0.8]),
            "PercentSalaryHike": np.random.randint(11, 25, size),
            "RelationshipSatisfaction": np.random.randint(1, 5, size),
            "StockOptionLevel": np.random.randint(0, 4, size),
            "TotalWorkingYears": np.random.randint(0, 5, size) if leaving else np.random.randint(3, 35, size),
            "TrainingTimesLastYear": np.random.randint(0, 7, size),
            "WorkLifeBalance": np.random.choice([1, 2], size) if leaving else np.random.choice([3, 4], size),
            "YearsAtCompany": np.random.randint(0, 3, size) if leaving else np.random.randint(1, 30, size),
            "YearsInCurrentRole": np.random.randint(0, 3, size) if leaving else np.random.randint(1, 15, size),
            "YearsSinceLastPromotion": np.random.randint(0, 10, size),
            "YearsWithCurrManager": np.random.randint(0, 3, size) if leaving else np.random.randint(1, 15, size),
        })
    df = pd.concat([make_group(n_leave, True), make_group(n - n_leave, False)]).sample(frac=1, random_state=42)
    return df.reset_index(drop=True)


@st.cache_data
def preprocess(df):
    df = df.copy()
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
    df["OverTime"]  = (df["OverTime"] == "Yes").astype(int)
    df["Gender"]    = (df["Gender"] == "Male").astype(int)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]
    return X, y


@st.cache_resource
def train(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_sc, X_te_sc = scaler.fit_transform(X_tr), scaler.transform(X_te)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=150, random_state=42),
    }
    results = {}
    for name, mdl in models.items():
        X_in_tr = X_tr_sc if name == "Logistic Regression" else X_tr
        X_in_te = X_te_sc if name == "Logistic Regression" else X_te
        mdl.fit(X_in_tr, y_tr)
        y_pred = mdl.predict(X_in_te)
        y_prob = mdl.predict_proba(X_in_te)[:, 1]
        results[name] = {
            "model": mdl, "Accuracy": accuracy_score(y_te, y_pred),
            "F1":    f1_score(y_te, y_pred),
            "AUC":   roc_auc_score(y_te, y_prob),
            "y_test": y_te, "y_pred": y_pred, "y_prob": y_prob,
        }
    return results, scaler, X_tr, X_te, y_tr, y_te


# ── Load data ─────────────────────────────────
if uploaded:
    raw_df = pd.read_csv(uploaded)
else:
    raw_df = generate_data()

X, y = preprocess(raw_df)
model_results, scaler, X_tr, X_te, y_tr, y_te = train(X, y)

# ═══════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════
st.title("👥 HR Employee Attrition Prediction System")
tabs = st.tabs(["📊 Data Overview", "📈 EDA", "🤖 Model Results", "🔍 Predict Employee"])

# ── TAB 1: Data Overview ──────────────────────
with tabs[0]:
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", len(raw_df))
    col2.metric("Attrition Count", (raw_df["Attrition"] == "Yes").sum())
    col3.metric("Attrition Rate", f"{(raw_df['Attrition']=='Yes').mean()*100:.1f}%")
    col4.metric("Features", raw_df.shape[1])

    st.dataframe(raw_df.head(50), use_container_width=True)
    st.write("**Data Types:**")
    st.dataframe(raw_df.dtypes.reset_index().rename(columns={"index": "Column", 0: "dtype"}), use_container_width=True)

# ── TAB 2: EDA ───────────────────────────────
with tabs[1]:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        counts = raw_df["Attrition"].value_counts()
        ax.bar(counts.index, counts.values, color=["#2ecc71", "#e74c3c"])
        ax.set_title("Attrition Distribution")
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots()
        raw_df.boxplot(column="Age", by="Attrition", ax=ax)
        plt.title("Age by Attrition"); plt.suptitle("")
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots()
        raw_df.boxplot(column="MonthlyIncome", by="Attrition", ax=ax)
        plt.title("Monthly Income by Attrition"); plt.suptitle("")
        st.pyplot(fig); plt.close()

    with col4:
        fig, ax = plt.subplots()
        ot = raw_df.groupby(["OverTime", "Attrition"]).size().unstack()
        ot.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
        ax.set_title("OverTime vs Attrition")
        st.pyplot(fig); plt.close()

    # Correlation heatmap
    st.subheader("Correlation Heatmap (Numeric Features)")
    numeric = raw_df.select_dtypes(include=np.number).copy()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(numeric.corr(), cmap="coolwarm", ax=ax, linewidths=0.3, annot=False)
    st.pyplot(fig); plt.close()

# ── TAB 3: Model Results ─────────────────────
with tabs[2]:
    st.subheader("Model Performance Comparison")

    summary = pd.DataFrame({
        name: {"Accuracy": v["Accuracy"], "F1-Score": v["F1"], "ROC-AUC": v["AUC"]}
        for name, v in model_results.items()
    }).T
    st.dataframe(summary.style.highlight_max(axis=0, color="#2ecc7155"), use_container_width=True)

    # Confusion Matrices
    st.subheader("Confusion Matrices")
    cols = st.columns(3)
    for i, (name, v) in enumerate(model_results.items()):
        with cols[i]:
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix(v["y_test"], v["y_pred"]),
                                   display_labels=["Stay", "Leave"]).plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(name)
            st.pyplot(fig); plt.close()

    # Feature Importance
    st.subheader("Top 20 Feature Importances (Random Forest)")
    rf = model_results["Random Forest"]["model"]
    imp = pd.Series(rf.feature_importances_, index=X_tr.columns).nlargest(20).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    imp.plot(kind="barh", ax=ax, color="#3498db")
    ax.set_xlabel("Importance")
    st.pyplot(fig); plt.close()

# ── TAB 4: Predict ───────────────────────────
with tabs[3]:
    st.subheader("🔍 Predict Attrition for a New Employee")

    c1, c2, c3 = st.columns(3)
    age         = c1.slider("Age", 18, 60, 30)
    income      = c2.number_input("Monthly Income ($)", 1000, 25000, 5000, step=500)
    yrs_company = c3.slider("Years at Company", 0, 40, 3)
    overtime    = c1.selectbox("OverTime", ["No", "Yes"])
    job_sat     = c2.selectbox("Job Satisfaction (1=Low, 4=High)", [1, 2, 3, 4], index=2)
    wlb         = c3.selectbox("Work-Life Balance (1=Low, 4=High)", [1, 2, 3, 4], index=2)
    distance    = c1.slider("Distance From Home (km)", 1, 30, 5)
    total_years = c2.slider("Total Working Years", 0, 40, 5)
    num_cos     = c3.slider("Num Companies Worked", 0, 9, 1)

    chosen_model = st.selectbox("Select Model", list(model_results.keys()))

    if st.button("🚀 Predict Attrition Risk"):
        raw_input = {col: 0 for col in X.columns}
        raw_input["Age"]                  = age
        raw_input["MonthlyIncome"]        = income
        raw_input["YearsAtCompany"]       = yrs_company
        raw_input["OverTime"]             = 1 if overtime == "Yes" else 0
        raw_input["JobSatisfaction"]      = job_sat
        raw_input["WorkLifeBalance"]      = wlb
        raw_input["DistanceFromHome"]     = distance
        raw_input["TotalWorkingYears"]    = total_years
        raw_input["NumCompaniesWorked"]   = num_cos

        mdl   = model_results[chosen_model]["model"]
        X_new = pd.DataFrame([raw_input])
        X_in  = scaler.transform(X_new) if chosen_model == "Logistic Regression" else X_new
        prob  = mdl.predict_proba(X_in)[0][1]

        st.divider()
        if prob >= 0.7:
            st.error(f"🔴 HIGH RISK — Attrition Probability: **{prob*100:.1f}%**")
        elif prob >= 0.4:
            st.warning(f"🟡 MEDIUM RISK — Attrition Probability: **{prob*100:.1f}%**")
        else:
            st.success(f"🟢 LOW RISK — Attrition Probability: **{prob*100:.1f}%**")

        # Gauge chart
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.barh(["Risk"], [prob], color="#e74c3c" if prob > 0.5 else "#2ecc71", height=0.4)
        ax.barh(["Risk"], [1 - prob], left=[prob], color="#ecf0f1", height=0.4)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_title(f"Attrition Risk: {prob*100:.1f}%")
        st.pyplot(fig); plt.close()

        st.markdown("**💡 Recommendations:**")
        recs = []
        if overtime == "Yes":       recs.append("- Reduce overtime workload")
        if job_sat <= 2:            recs.append("- Improve job satisfaction (growth opportunities, recognition)")
        if wlb <= 2:                recs.append("- Improve work-life balance (flexible hours)")
        if income < 3000:           recs.append("- Review compensation — salary is below average")
        if distance > 20:           recs.append("- Consider remote/hybrid options (long commute)")
        if not recs:                recs.append("- Employee appears engaged. Maintain current conditions.")
        for r in recs:
            st.write(r)
