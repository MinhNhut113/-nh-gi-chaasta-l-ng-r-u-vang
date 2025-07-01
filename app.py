import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

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

# Tạo các input cho người dùng nhập
fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.4)
volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.7)
citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.0)
residual_sugar = st.number_input("Residual Sugar", 0.0, 15.0, 1.9)
chlorides = st.number_input("Chlorides", 0.0, 0.2, 0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0, 80.0, 11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0, 300.0, 34.0)
density = st.number_input("Density", 0.9900, 1.0050, 0.9978)
pH = st.number_input("pH", 2.5, 4.5, 3.51)
sulphates = st.number_input("Sulphates", 0.3, 2.0, 0.56)
alcohol = st.number_input("Alcohol", 8.0, 15.0, 9.4)

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
