import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Rainfall Prediction System",
    page_icon="🌧",
    layout="centered"
)

# Title
st.title("🌧 Rainfall Prediction System")
st.caption("Machine Learning based rainfall analysis and prediction")

# Load model and feature columns
model = joblib.load("model/rainfall_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

# Load dataset
df = pd.read_csv("data/rainfall.csv")

subdivisions = sorted(df["SUBDIVISION"].unique())
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# Tabs
tab1, tab2 = st.tabs([
    "📅 Annual Rainfall Prediction",
    "📊 Monthly Rainfall Insights"
])

# ================= TAB 1 =================
with tab1:
    st.subheader("📅 Annual Rainfall Prediction")

    subdivision = st.selectbox("Select Subdivision", subdivisions)

    st.markdown("### Enter Monthly Rainfall (mm)")

    monthly_values = {}
    col1, col2, col3 = st.columns(3)

    for i, month in enumerate(months):
        with [col1, col2, col3][i % 3]:
            monthly_values[month] = st.number_input(
                month, min_value=0.0, value=50.0
            )

    # Prepare input
    input_data = {
        "SUBDIVISION": subdivision,
        **monthly_values
    }

    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)

    # Align columns
    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    if st.button("🔮 Predict Annual Rainfall"):
        prediction = model.predict(input_encoded)[0]
        st.success(f"🌧 Predicted Annual Rainfall: **{prediction:.2f} mm**")

# ================= TAB 2 =================
with tab2:
    st.subheader("📊 Monthly Rainfall Insights (Historical Data)")

    colA, colB = st.columns(2)

    with colA:
        selected_sub = st.selectbox(
            "Select Subdivision",
            subdivisions,
            key="monthly_sub"
        )

    with colB:
        selected_month = st.selectbox(
            "Select Month",
            months,
            key="monthly_month"
        )

    filtered = df[df["SUBDIVISION"] == selected_sub]

    avg_rain = filtered[selected_month].mean()
    min_rain = filtered[selected_month].min()
    max_rain = filtered[selected_month].max()

    st.metric("🌦 Average Rainfall (mm)", f"{avg_rain:.2f}")
    st.metric("⬇ Minimum Recorded (mm)", f"{min_rain:.2f}")
    st.metric("⬆ Maximum Recorded (mm)", f"{max_rain:.2f}")

    st.markdown("### 📈 Monthly Rainfall Trend (All Months)")
    st.line_chart(filtered[months])
