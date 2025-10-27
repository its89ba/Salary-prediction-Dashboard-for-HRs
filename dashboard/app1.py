import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =======================
# Page Configuration
# =======================
st.set_page_config(
    page_title="💼 HR Salary Prediction Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =======================
# Custom CSS Styling
# =======================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =======================
# Load Resources
# =======================
@st.cache_resource
def load_model():
    return joblib.load("gradient_boosting_model.pkl")

@st.cache_data
def load_data():
    return pd.read_excel("salary_dataset.xlsx")

try:
    model = load_model()
    data = load_data()
except Exception as e:
    st.error(f"❌ Error loading resources: {e}")
    st.stop()

# =======================
# Header Section
# =======================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">💼 HR Salary Prediction Dashboard</h1>', 
                unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-bottom: 3rem;'>
    <h3 style='color: #666; font-weight: 300;'>
    Predict employee salaries based on <strong>Age</strong> and <strong>Years of Experience</strong>
    </h3>
    <p>Powered by Gradient Boosting Regression 🤖</p>
</div>
""", unsafe_allow_html=True)

# =======================
# Sidebar for Inputs
# =======================
with st.sidebar:
    st.markdown("## 🧍 Employee Details")
    
    with st.container():
        st.markdown("### Basic Information")
        age = st.slider("**Age**", min_value=18, max_value=65, value=30, 
                       help="Select the employee's age")
        experience = st.slider("**Years of Experience**", min_value=0, max_value=50, value=5,
                              help="Select years of professional experience")
    
    st.markdown("---")
    
    # Additional features
    st.markdown("### 📊 Model Information")
    st.metric("Training Samples", f"{len(data):,}")
    st.metric("Features", "2 (Age, Experience)")
    
    st.markdown("---")
    predict_btn = st.button("🎯 Predict Salary", use_container_width=True)

# =======================
# Main Content Area
# =======================
if predict_btn:
    # Prepare input and make prediction
    input_data = np.array([[age, experience]])
    prediction = model.predict(input_data)[0]
    
    # Display prediction in a nice card
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class='prediction-card'>
            <h2 style='text-align: center; color: #1f77b4;'>💰 Predicted Salary</h2>
            <h1 style='text-align: center; color: #2ecc71; font-size: 3rem;'>${prediction:,.2f}</h1>
            <p style='text-align: center; color: #666;'>Based on Age: {age} | Experience: {experience} years</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Salary Distribution")
        
        # Create interactive plot with Plotly
        fig1 = px.scatter(data, x="Year of Experience", y="Current Salary", 
                         hover_data=["Age"],
                         title="Salary vs Experience",
                         color_discrete_sequence=['#1f77b4'])
        
        # Add prediction point
        fig1.add_trace(go.Scatter(
            x=[experience], y=[prediction],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='Predicted Employee'
        ))
        
        fig1.update_layout(
            height=400,
            showlegend=True,
            xaxis_title="Years of Experience",
            yaxis_title="Salary ($)"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("📊 Age vs Salary")
        
        fig2 = px.scatter(data, x="Age", y="Current Salary",
                         hover_data=["Year of Experience"],
                         title="Salary vs Age",
                         color_discrete_sequence=['#ff7f0e'])
        
        fig2.add_trace(go.Scatter(
            x=[age], y=[prediction],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='Predicted Employee'
        ))
        
        fig2.update_layout(
            height=400,
            showlegend=True,
            xaxis_title="Age",
            yaxis_title="Salary ($)"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Statistics Section
    st.subheader("📋 Quick Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Dataset Mean Salary", f"${data['Current Salary'].mean():,.2f}")
    with col2:
        st.metric("Dataset Max Salary", f"${data['Current Salary'].max():,.2f}")
    with col3:
        st.metric("Predicted Salary", f"${prediction:,.2f}")
    with col4:
        diff_percent = ((prediction - data['Current Salary'].mean()) / data['Current Salary'].mean()) * 100
        st.metric("Vs Mean", f"{diff_percent:+.1f}%")

# =======================
# Dataset Preview Section
# =======================
st.markdown("---")
with st.expander("📁 Dataset Overview", expanded=False):
    tab1, tab2, tab3 = st.tabs(["📊 Preview", "📈 Statistics", "🔍 Data Info"])
    
    with tab1:
        st.subheader("First 10 Records")
        st.dataframe(data.head(10), use_container_width=True)
    
    with tab2:
        st.subheader("Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)
    
    with tab3:
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", len(data))
            st.metric("Columns", len(data.columns))
        with col2:
            st.metric("Data Types", str(data.dtypes.tolist()))
            st.metric("Missing Values", data.isnull().sum().sum())

# =======================
# Footer
# =======================
st.markdown("""
---
<div style='text-align: center; color: #666;'>
    <p>🔹 Developed by Hiba Ali | 💡 Powered by Gradient Boosting Regression</p>
    <p style='font-size: 0.8rem;'>HR Analytics Dashboard v1.0</p>
</div>
""", unsafe_allow_html=True)