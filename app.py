
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.title("🚀 Credit Intelligence System")

df = pd.read_csv("data.csv")

# Encoding
le = LabelEncoder()
df["Employment"] = le.fit_transform(df["Employment"])
df["LoanStatus"] = df["LoanStatus"].map({"Approved":1,"Rejected":0})

X = df.drop("LoanStatus", axis=1)
y = df["LoanStatus"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

st.header("📊 Model Performance")

for name, model in models.items():
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test,y_pred)
    pre = precision_score(y_test,y_pred)
    rec = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)

    st.subheader(name)
    st.write({"Accuracy":acc,"Precision":pre,"Recall":rec,"F1":f1})

    # ROC
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test,y_prob)
    fig = px.line(x=fpr, y=tpr, title=f"ROC Curve - {name}")
    st.plotly_chart(fig)

# Clustering
st.header("🧠 Customer Segmentation")
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

fig = px.scatter(df, x="Income", y="Debt", color="Cluster")
st.plotly_chart(fig)

# Association Rules
st.header("🔗 Association Rules")

df_rules = pd.get_dummies(df[["Risk","LoanStatus"]])
freq = apriori(df_rules, min_support=0.1, use_colnames=True)
rules = association_rules(freq, metric="confidence", min_threshold=0.6)

st.write(rules[["antecedents","consequents","confidence","lift"]].head())

# Prediction
st.header("🔮 New Customer Prediction")

income = st.number_input("Income",20000,100000)
debt = st.number_input("Debt",0,100000)
emi = st.slider("EMI",0,100)
late = st.slider("Late Payments",0,10)
emp = st.selectbox("Employment",["Salaried","SelfEmployed"])

emp_val = le.transform([emp])[0]

input_df = pd.DataFrame([[income,debt,emi,late,emp_val,700,1]],
columns=X.columns)

model = RandomForestClassifier()
model.fit(X,y)

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    st.success("Approved" if pred==1 else "Rejected")

# Upload new data
st.header("📂 Upload New Data")
file = st.file_uploader("Upload CSV")

if file:
    new_df = pd.read_csv(file)
    st.write(new_df.head())
