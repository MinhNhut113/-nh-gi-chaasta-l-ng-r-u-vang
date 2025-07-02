import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load dữ liệu và train model
@st.cache_data
def train_model():
    data = pd.read_csv('winequality-red.csv')
    X = data.drop('quality', axis=1)
    y = data['quality']
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model()

# Giao diện
st.set_page_config(page_title="Wine Quality Prediction", layout="centered")
st.title("🍷 Dự đoán chất lượng rượu vang")
st.write("""
Dựa trên các chỉ số hóa học để dự đoán điểm chất lượng (0-10) của rượu vang.
""")

# Các input không giới hạn (dùng min/max rộng)
fixed_acidity = st.number_input("Fixed Acidity", min_value=-1e10, max_value=1e10)
volatile_acidity = st.number_input("Volatile Acidity", min_value=-1e10, max_value=1e10)
citric_acid = st.number_input("Citric Acid", min_value=-1e10, max_value=1e10)
residual_sugar = st.number_input("Residual Sugar", min_value=-1e10, max_value=1e10)
chlorides = st.number_input("Chlorides", min_value=-1e10, max_value=1e10)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=-1e10, max_value=1e10)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=-1e10, max_value=1e10)
density = st.number_input("Density", min_value=-1e10, max_value=1e10)
pH = st.number_input("pH", min_value=-1e10, max_value=1e10)
sulphates = st.number_input("Sulphates", min_value=-1e10, max_value=1e10)
alcohol = st.number_input("Alcohol", min_value=-1e10, max_value=1e10)

if st.button("Dự đoán chất lượng"):
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                          free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    prediction = model.predict(features)[0]
    prediction = round(prediction, 2)
    
    st.success(f"🎯 Điểm chất lượng dự đoán: **{prediction} / 10**")
    
    if prediction >= 7:
        st.markdown("✅ Đây là rượu vang **chất lượng cao**!")
    elif prediction >= 5:
        st.markdown("⚠️ Đây là rượu vang **trung bình**.")
    else:
        st.markdown("🚫 Đây là rượu vang **chất lượng thấp**.")
