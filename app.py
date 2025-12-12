import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Rainfall Prediction System",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0052a3;
        transform: scale(1.05);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 20px 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        text-align: center;
        color: #0066cc;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)


# Load the model and feature names
@st.cache_resource
def load_model():
    try:
        with open('rainfall_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, features
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found! Please ensure files are in the project directory.")
        st.stop()


model, feature_names = load_model()

# Header Section
st.markdown("<h1>🌧️ Rainfall Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter weather parameters to predict rainfall probability using Machine Learning</p>",
            unsafe_allow_html=True)
st.markdown("---")

# Sidebar for information
with st.sidebar:
    st.header("ℹ️ About the Model")
    st.info("""
    This machine learning model predicts rainfall based on various weather parameters.

    **Model Details:**
    - Algorithm: Random Forest Classifier
    - Training Data: Weather dataset
    - Features: Climate and weather indicators
    """)

    st.header("📊 How to Use")
    st.markdown("""
    1. Enter values for each weather parameter
    2. Click **'Predict Rainfall'** button
    3. View prediction results and confidence level
    4. Get weather recommendations
    """)

    st.markdown("---")

    # Option to view sample data
    if st.checkbox("📂 Show Sample Data"):
        try:
            df = pd.read_csv('Rainfall.csv')
            st.dataframe(df.head(), use_container_width=True)
        except:
            st.warning("Dataset file not found")

    st.markdown("---")
    st.markdown("**💻 Built with Streamlit & scikit-learn**")
    st.markdown("**👨‍💻 ML Engineer Project**")

# Main content - Input Section
st.subheader("📝 Enter Weather Parameters")
st.markdown("Please provide the following weather data for prediction:")

# Create dynamic columns based on number of features (3 columns layout)
num_cols = 3
rows = (len(feature_names) + num_cols - 1) // num_cols

inputs = {}
for i in range(rows):
    cols = st.columns(num_cols)
    for j in range(num_cols):
        idx = i * num_cols + j
        if idx < len(feature_names):
            feature = feature_names[idx]
            with cols[j]:
                # Format feature name nicely
                display_name = feature.replace('_', ' ').title()

                # Create input fields with appropriate ranges
                inputs[feature] = st.number_input(
                    f"🔹 {display_name}",
                    value=0.0,
                    step=0.1,
                    format="%.2f",
                    help=f"Enter the value for {display_name}",
                    key=feature
                )

st.markdown("---")

# Prediction Section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 Predict Rainfall", use_container_width=True)

if predict_button:
    # Validate inputs
    if all(value == 0.0 for value in inputs.values()):
        st.warning("⚠️ Please enter weather parameter values before prediction!")
    else:
        with st.spinner("🔄 Analyzing weather data..."):
            # Create dataframe from inputs
            input_df = pd.DataFrame([inputs])

            try:
                # Make prediction
                prediction = model.predict(input_df)
                prediction_proba = model.predict_proba(input_df)

                # Get confidence
                confidence = max(prediction_proba[0]) * 100
                rain_prob = prediction_proba[0][1] * 100 if len(prediction_proba[0]) > 1 else 0
                no_rain_prob = prediction_proba[0][0] * 100

                # Display results with better styling
                st.markdown("---")
                st.subheader("📈 Prediction Results")

                # Create metrics columns
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    if prediction[0] == 1:
                        st.metric(
                            label="🌧️ Prediction",
                            value="Rain Expected",
                            delta="High Alert" if confidence > 75 else "Moderate"
                        )
                    else:
                        st.metric(
                            label="☀️ Prediction",
                            value="No Rain",
                            delta="Clear Skies"
                        )
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="🎯 Confidence Level",
                        value=f"{confidence:.1f}%",
                        delta="High Confidence" if confidence > 70 else "Moderate Confidence"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    if prediction[0] == 1:
                        st.metric(
                            label="💡 Recommendation",
                            value="Carry Umbrella",
                            delta="☂️"
                        )
                    else:
                        st.metric(
                            label="💡 Recommendation",
                            value="Enjoy Weather",
                            delta="😎"
                        )
                    st.markdown('</div>', unsafe_allow_html=True)

                # Detailed probability breakdown
                st.markdown("---")
                st.subheader("📊 Probability Distribution")

                col1, col2 = st.columns([2, 1])

                with col1:
                    # Create probability dataframe
                    prob_df = pd.DataFrame({
                        'Outcome': ['No Rain ☀️', 'Rain 🌧️'],
                        'Probability (%)': [no_rain_prob, rain_prob]
                    })

                    # Display bar chart
                    st.bar_chart(prob_df.set_index('Outcome'))

                with col2:
                    st.markdown("### Breakdown")
                    st.metric("No Rain", f"{no_rain_prob:.1f}%")
                    st.metric("Rain", f"{rain_prob:.1f}%")

                # Success/Info message with interpretation
                st.markdown("---")
                if prediction[0] == 1:
                    st.success(f"🌧️ **Rain is Predicted** with {confidence:.1f}% confidence!")
                    st.info("""
                    **What this means:**
                    - There's a high likelihood of rainfall
                    - Consider carrying rain gear
                    - Plan indoor activities if possible
                    - Check weather updates regularly
                    """)
                else:
                    st.success(f"☀️ **No Rain Predicted** with {confidence:.1f}% confidence!")
                    st.info("""
                    **What this means:**
                    - Clear weather is expected
                    - Good for outdoor activities
                    - Low chance of precipitation
                    - Enjoy the day!
                    """)

                # Display input summary
                with st.expander("📋 View Input Parameters"):
                    input_summary = pd.DataFrame([inputs]).T
                    input_summary.columns = ['Value']
                    input_summary.index = [name.replace('_', ' ').title() for name in input_summary.index]
                    st.dataframe(input_summary, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.info("Please check if your input values are valid.")

# Additional Features Section
st.markdown("---")
st.subheader("📚 Additional Features")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 View Model Info", use_container_width=True):
        st.info(f"""
        **Model Information:**
        - Total Features: {len(feature_names)}
        - Algorithm: Random Forest Classifier
        - Output Classes: Rain / No Rain
        """)

with col2:
    if st.button("🔄 Reset Form", use_container_width=True):
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p>© 2025 Rainfall Prediction System | Powered by Machine Learning & Streamlit</p>
        <p>Developed as part of AI/ML Engineering Project</p>
    </div>
    """, unsafe_allow_html=True)
