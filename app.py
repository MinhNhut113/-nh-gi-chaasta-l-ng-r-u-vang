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
    model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, epsilon=0.1))
    model.fit(X, y)
    return model

model = train_svm_model()

# Thiết lập page
st.set_page_config(page_title="Wine Quality Prediction (SVM)", layout="wide")

# CSS tùy biến
st.markdown("""
<style>
body { 
    font-family: 'Raleway', sans-serif; 
    background-color: #f8f3ed; 
    font-size: 15px;
}
h1, h2 { 
    font-family: 'Playfair Display', serif; 
}
.header-font {
    font-family: 'Playfair Display', serif;
}
.wine-red {
    color: #722f37; 
}
.bg-wine {
    color: white;
    padding: 20px;
    border-radius: 8px;
}
.form-box { 
    background-color: #722f37; 
    color: white;
    padding: 20px; 
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
.result-box { 
    background-color: #722f37;
    color: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
.progress {
    height: 16px;
    background-color: #e2e8f0; 
    border-radius: 8px;
    overflow: hidden; 
}
.progress-bar { 
    height: 100%;
    background-color: #f8f3ed;
    width: 0%;
    transition: width 0.5s; 
}
</style>
""", unsafe_allow_html=True)

# Tiêu đề
st.markdown('<div class="bg-wine"><h1 class="header-font" style="text-align:center;">🍷 Dự đoán chất lượng rượu vang (SVM)</h1></div>', unsafe_allow_html=True)

# INPUT WINE PARAMETERS
st.markdown('<div class="form-box"><h2 class="wine-red header-font">Input Wine Parameters</h2>', unsafe_allow_html=True)

def float_input(label, key):
    value = st.text_input(label, key=key, placeholder="Enter value...")
    try:
        return float(value)
    except:
        return np.nan

col1, col2 = st.columns(2)
with col1:
    fixed_acidity = float_input("Fixed Acidity", "fixed_acidity")
    citric_acid = float_input("Citric Acid", "citric_acid")
    chlorides = float_input("Chlorides", "chlorides")
    total_sulfur_dioxide = float_input("Total Sulfur Dioxide", "total_sulfur_dioxide")
    pH = float_input("pH", "pH")
    alcohol = float_input("Alcohol", "alcohol")

with col2:
    volatile_acidity = float_input("Volatile Acidity", "volatile_acidity")
    residual_sugar = float_input("Residual Sugar", "residual_sugar")
    free_sulfur_dioxide = float_input("Free Sulfur Dioxide", "free_sulfur_dioxide")
    density = float_input("Density", "density")
    sulphates = float_input("Sulphates", "sulphates")

predict_btn = st.button("Predict Wine Quality")
st.markdown('</div>', unsafe_allow_html=True)

# KẾT QUẢ DỰ ĐOÁN
st.markdown('<div class="result-box"><h2 class="wine-red header-font">Prediction Results</h2>', unsafe_allow_html=True)
if predict_btn:
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                          residual_sugar, chlorides, free_sulfur_dioxide,
                          total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    if np.isnan(features).any():
        st.warning("⚠️ Vui lòng nhập đầy đủ tất cả thông tin rượu.")
    else:
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
            st.success("✅ Đây là rượu vang chất lượng cao!")
        elif prediction >= 5:
            st.warning("⚠️ Đây là rượu vang trung bình.")
        else:
            st.error("🚫 Đây là rượu vang chất lượng thấp.")
else:
    st.info("Nhập thông tin rượu và bấm Predict để xem kết quả.")
st.markdown('</div>', unsafe_allow_html=True)
