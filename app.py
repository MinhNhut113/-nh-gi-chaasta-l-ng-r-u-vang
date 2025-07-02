import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load và train model
@st.cache_data
def train_model():
    data = pd.read_csv('winequality-red.csv')
    X = data.drop('quality', axis=1)
    y = data['quality'].apply(lambda x: 1 if x >=7 else 0)  # Chất lượng cao >=7
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = train_model()

# Giao diện Streamlit
st.set_page_config(page_title="Wine Quality SVM Prediction", layout="centered")
st.title("🍷 Dự đoán chất lượng rượu vang bằng SVM")
st.write("Nhập các chỉ số hóa học để dự đoán khả năng rượu vang chất lượng cao (>=7).")

# Tạo input (trống)
fixed_acidity = st.text_input("Fixed Acidity")
volatile_acidity = st.text_input("Volatile Acidity")
citric_acid = st.text_input("Citric Acid")
residual_sugar = st.text_input("Residual Sugar")
chlorides = st.text_input("Chlorides")
free_sulfur_dioxide = st.text_input("Free Sulfur Dioxide")
total_sulfur_dioxide = st.text_input("Total Sulfur Dioxide")
density = st.text_input("Density")
pH = st.text_input("pH")
sulphates = st.text_input("Sulphates")
alcohol = st.text_input("Alcohol")

if st.button("Dự đoán"):
    try:
        # Ép kiểu float
        features = np.array([[
            float(fixed_acidity), float(volatile_acidity), float(citric_acid),
            float(residual_sugar), float(chlorides), float(free_sulfur_dioxide),
            float(total_sulfur_dioxide), float(density), float(pH),
            float(sulphates), float(alcohol)
        ]])
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1] * 100
        
        if prediction == 1:
            st.success(f"✅ Đây có thể là rượu vang **chất lượng cao** (≥7) với xác suất khoảng **{probability:.1f}%**.")
        else:
            st.warning(f"⚠️ Đây có thể **không phải rượu vang chất lượng cao** (xác suất chỉ khoảng **{probability:.1f}%**).")

    except ValueError:
        st.error("❌ Vui lòng nhập đầy đủ và chính xác các chỉ số.")

