import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

@st.cache_data
def train_svm_model():
    data = pd.read_csv('winequality-red.csv')
    X = data.drop('quality', axis=1)
    y = data['quality']
    # pipeline: chuẩn hóa + SVR
    model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, epsilon=0.1))
    model.fit(X, y)
    return model

model = train_svm_model()

st.set_page_config(page_title="Wine Quality Prediction (SVM)", layout="wide")

st.markdown("""
<style>
body { font-family: 'Raleway', sans-serif; background-color: #f8f3ed; font-size: 15px; }
h1, h2 { font-family: 'Playfair Display', serif; }
.header-font { font-family: 'Playfair Display', serif; }
.wine-red { color: #722f37; }
.bg-wine { background-color: #722f37; color: white; padding: 20px; border-radius: 8px; }
.form-box { background-color: #722f37; color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
.result-box { background-color: #722f37; color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
.progress { height: 16px; background-color: #e2e8f0; border-radius: 8px; overflow: hidden; }
.progress-bar { height: 100%; background-color: #f8f3ed; width: 0%; transition: width 0.5s; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="bg-wine"><h1 class="header-font" style="text-align:center;">Dự Đoán Chất Lượng Rượu Vang (SVM)</h1></div>', unsafe_allow_html=True)

st.markdown('<div class="form-box"><h2 class="wine-red header-font">Input Wine Parameters</h2>', unsafe_allow_html=True)

# Input theo 2 cột
col1, col2 = st.columns(2)
with col1:
    fixed_acidity = st.text_input("Fixed Acidity")
    citric_acid = st.text_input("Citric Acid")
    chlorides = st.text_input("Chlorides")
    total_sulfur_dioxide = st.text_input("Total Sulfur Dioxide")
    pH = st.text_input("pH")
    alcohol = st.text_input("Alcohol")

with col2:
    volatile_acidity = st.text_input("Volatile Acidity")
    residual_sugar = st.text_input("Residual Sugar")
    free_sulfur_dioxide = st.text_input("Free Sulfur Dioxide")
    density = st.text_input("Density")
    sulphates = st.text_input("Sulphates")

predict_btn = st.button("Predict Wine Quality")
st.markdown('</div>', unsafe_allow_html=True)

# Kết quả
st.markdown('<div class="result-box"><h2 class="wine-red header-font">Prediction Results</h2>', unsafe_allow_html=True)

if predict_btn:
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

        st.markdown(f"""
            <div class="wine-red" style="font-size: 24px; font-weight: bold;">
                Quality Score (0-10): {prediction}
            </div>
            <div class="progress">
                <div class="progress-bar" style="width: {min(100, max(0, prediction/10*100))}%"></div>
            </div>
        """, unsafe_allow_html=True)

        if prediction >= 7:
            st.success("✅ Excellent quality wine!")
            st.image("https://cdn-icons-png.flaticon.com/512/979/979585.png", width=100)
        elif prediction >= 5:
            st.warning("⚠️ Good quality wine.")
            st.image("https://cdn-icons-png.flaticon.com/512/5793/5793147.png", width=100)
        else:
            st.error("🚫 Low quality wine.")
            st.image("https://cdn-icons-png.flaticon.com/512/7556/7556216.png", width=100)
    except ValueError:
        st.error("❌ Vui lòng nhập đầy đủ và chính xác tất cả các chỉ số.")
else:
    st.info("Nhập thông tin rượu và bấm Predict để xem kết quả.")
st.markdown('</div>', unsafe_allow_html=True)
