import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Wine Quality Prediction", layout="wide")
st.title("🍷 Wine Quality Prediction & EDA App")

# Load data
data = pd.read_csv('winequality-red.csv', sep=';')

# Sidebar
st.sidebar.header("Settings")
show_data = st.sidebar.checkbox("Show raw data", True)
selected_model = st.sidebar.selectbox("Select model to train", 
                                      ["Logistic Regression", "Decision Tree", "Random Forest"])
test_size = st.sidebar.slider("Test size (%)", 10, 50, 30, step=5) / 100

# Show data
if show_data:
    st.subheader("Raw Dataset")
    st.dataframe(data.head())

# Data exploration
st.subheader("Target distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x='quality', data=data, ax=ax1)
st.pyplot(fig1)

# Heatmap correlation
st.subheader("Feature correlation heatmap")
fig2, ax2 = plt.subplots(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax2)
st.pyplot(fig2)

# Barplot alcohol vs quality
st.subheader("Average Alcohol by Quality")
fig3, ax3 = plt.subplots()
sns.barplot(x='quality', y='alcohol', data=data, ax=ax3)
st.pyplot(fig3)

# Preprocessing
data['quality'] = data['quality'].apply(lambda x: 1 if x >=7 else 0)
X = data.drop('quality', axis=1)
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Model training
st.subheader(f"Training Model: {selected_model}")

if selected_model == "Logistic Regression":
    model = LogisticRegression()
elif selected_model == "Decision Tree":
    model = DecisionTreeClassifier()
else:
    model = RandomForestClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"**Accuracy:** {acc*100:.2f}%")

# Classification report
st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion matrix
st.subheader("Confusion Matrix")
fig4, ax4 = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax4)
st.pyplot(fig4)

# TP TN FP FN
cm = confusion_matrix(y_test, y_pred)
try:
    st.write(f"TN: {cm[0][0]} | FP: {cm[0][1]}")
    st.write(f"FN: {cm[1][0]} | TP: {cm[1][1]}")
except:
    pass
