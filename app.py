import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load dữ liệu và train mô hình SVM
@st.cache_data
def train_svm_model():
    data = pd.read_csv('winequality-red.csv')
    X = data.drop('quality', axis=1)
    y = data['quality']
    # Dùng pipeline chuẩn hóa rồi SVR
    model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, epsilon=0.1))
    model.fit(X, y)
    return model

model = train_svm_model()

# Giao diện
st.set_page_config(page_title="Wine Quality Prediction (SVM)", layout="centered")
st.title("🍷 Dự đoán chất lượng rượu vang")
st.write("""
Dựa trên các chỉ số hóa học để dự đoán điểm chất lượng (0-10) của rượu.
""")

# Input không có giá trị mặc định
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

if st.button("Dự đoán chất lượng"):
    try:
        features = np.array([[
            float(fixed_acidity),
            float(volatile_acidity),
            float(citric_acid),
            float(residual_sugar),
            float(chlorides),
            float(free_sulfur_dioxide),
            float(total_sulfur_dioxide),
            float(density),
            float(pH),
            float(sulphates),
            float(alcohol)
        ]])
        
        prediction = model.predict(features)[0]
        prediction = round(prediction, 2)
        
        st.success(f"🎯 Điểm chất lượng dự đoán: **{prediction} / 10**")
        if prediction >= 7:
            st.markdown("✅ Đây là rượu vang **chất lượng cao**!")
        elif prediction >= 5:
            st.markdown("⚠️ Đây là rượu vang **trung bình**.")
        else:
            st.markdown("🚫 Đây là rượu vang **chất lượng thấp**.")

    except ValueError:
        st.error("❌ Vui lòng nhập đầy đủ và chính xác tất cả các chỉ số.")

