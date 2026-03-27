import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.cluster import KMeans

from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Credit Intelligence System", layout="wide")

st.title("🚀 Credit Intelligence System")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("data.csv")

st.subheader("📊 Raw Dataset")
st.dataframe(df.head())

# -------------------------------
# PREPROCESSING
# -------------------------------
df_encoded = pd.get_dummies(df, drop_first=True)

# Target column fix
if "LoanStatus_Approved" in df_encoded.columns:
    target = "LoanStatus_Approved"
else:
    st.error("Target column not found!")
    st.stop()

X = df_encoded.drop(target, axis=1)
y = df_encoded[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODEL TRAINING
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear'),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

st.header("📈 Model Performance")

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.subheader(name)
        st.write({
            "Accuracy": round(acc, 3),
            "Precision": round(pre, 3),
            "Recall": round(rec, 3),
            "F1 Score": round(f1, 3)
        })

        # ROC Curve
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            fig = px.line(
                x=fpr,
                y=tpr,
                title=f"ROC Curve - {name} (AUC={round(roc_auc,2)})"
            )
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"{name} failed: {e}")

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
st.header("🔥 Feature Importance (Random Forest)")

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig = px.bar(importance.head(10), x="Importance", y="Feature", orientation='h')
st.plotly_chart(fig)

# -------------------------------
# CLUSTERING
# -------------------------------
st.header("🧠 Customer Segmentation")

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

df["Cluster"] = clusters

fig = px.scatter(df, x="Income", y="Debt", color=df["Cluster"].astype(str))
st.plotly_chart(fig)

# -------------------------------
# ASSOCIATION RULES
# -------------------------------
st.header("🔗 Association Rules")

try:
    df_rules = pd.get_dummies(df[["Risk", "LoanStatus"]])
    freq = apriori(df_rules, min_support=0.1, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.6)

    st.dataframe(rules[["antecedents", "consequents", "confidence", "lift"]])

except Exception as e:
    st.error(f"Association rules error: {e}")

# -------------------------------
# PREDICTION SECTION
# -------------------------------
st.header("🔮 Predict New Customer")

income = st.number_input("Income", 20000, 100000, 40000)
debt = st.number_input("Debt", 0, 100000, 20000)
emi = st.slider("EMI (%)", 0, 100, 30)
late = st.slider("Late Payments", 0, 10, 1)
emp = st.selectbox("Employment", ["Salaried", "SelfEmployed"])

input_dict = {
    "Income": income,
    "Debt": debt,
    "EMI": emi,
    "Late": late,
    "Employment": emp,
    "CreditScore": 700,
    "Risk": "Medium"
}

input_df = pd.DataFrame([input_dict])

# Match encoding
input_encoded = pd.get_dummies(input_df)

# Align columns
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

# Train final model
final_model = RandomForestClassifier()
final_model.fit(X, y)

if st.button("Predict"):
    pred = final_model.predict(input_encoded)[0]
    prob = final_model.predict_proba(input_encoded)[0][1]

    if pred == 1:
        st.success(f"✅ Approved (Confidence: {round(prob,2)})")
    else:
        st.error(f"❌ Rejected (Confidence: {round(prob,2)})")

# -------------------------------
# FILE UPLOAD
# -------------------------------
st.header("📂 Upload New Data")

file = st.file_uploader("Upload CSV for Batch Prediction")

if file:
    new_df = pd.read_csv(file)
    new_encoded = pd.get_dummies(new_df)

    new_encoded = new_encoded.reindex(columns=X.columns, fill_value=0)

    preds = final_model.predict(new_encoded)

    new_df["Prediction"] = np.where(preds == 1, "Approved", "Rejected")

    st.dataframe(new_df.head())