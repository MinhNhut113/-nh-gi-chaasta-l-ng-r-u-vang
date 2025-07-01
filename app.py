import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os

# ---------------------------
# TRAIN MODEL & SAVE
# ---------------------------
# Nếu chưa có file model.pkl thì sẽ train và lưu
if not os.path.exists('model.pkl'):
    try:
        data = pd.read_csv('winequality-red.csv')
        st.write("✅ Đã đọc file CSV và bắt đầu huấn luyện mô hình...")
        X = data.drop('quality', axis=1)
        y = data['quality']

        model = LinearRegression()
        model.fit(X, y)

        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        st.write("✅ Mô hình đã huấn luyện và lưu vào model.pkl")
    except Exception as e:
        st.error(f"❌ Lỗi khi đọc CSV hoặc train model: {e}")
        st.stop()

# ---------------------------
# LOAD MODEL
# ---------------------------
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# ---------------------------
# STREAMLIT APP
# ---------------------------
st.set_page_config(page_title="Wine Quality Prediction", layout="centered")
st.title("🍷 Wine Quality Prediction App")

st.markdown("""
Nhập các chỉ số hóa học để dự đoán chất lượng rượu (0-10).
""")

# Các input để người dùng nhập
fixed_acidity = st.number_input("Fixed Acidity")
volatile_acidity = st.number_input("Volatile Acidity")
citric_acid = st.number_input("Citric Acid")
residual_sugar = st.number_input("Residual Sugar")
chlorides = st.number_input("Chlorides")
free_so2 = st.number_input("Free Sulfur Dioxide")
total_so2 = st.number_input("Total Sulfur Dioxide")
density = st.number_input("Density", format="%.5f")
pH = st.number_input("pH")
sulphates = st.number_input("Sulphates")
alcohol = st.number_input("Alcohol")

if st.button("Predict Wine Quality"):
    # Dự đoán
    input_features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                chlorides, free_so2, total_so2, density, pH, sulphates, alcohol]])
    prediction = model.predict(input_features)[0]
    prediction = round(prediction, 2)

    # Hiển thị kết quả
    st.subheader(f"Predicted Wine Quality: **{prediction}** (0-10)")
    if prediction >= 7:
        st.success("🎉 Excellent wine!")
    elif prediction >= 5:
        st.info("🙂 Good quality wine.")
    else:
        st.warning("⚠️ Average or below quality.")
