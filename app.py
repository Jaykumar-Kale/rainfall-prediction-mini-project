import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(
    page_title="Rainfall Prediction System",
    page_icon="🌧",
    layout="centered"
)

st.title("🌧 Rainfall Prediction System")
st.caption("Machine Learning based rainfall analysis and prediction")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/rainfall.csv")
    df.dropna(inplace=True)
    return df

df = load_data()

months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

subdivisions = sorted(df["SUBDIVISION"].unique())

# Train model once
@st.cache_resource
def train_model(data):
    X = data.drop("ANNUAL", axis=1)
    y = data["ANNUAL"]
    X = pd.get_dummies(X, drop_first=True)

    model = LinearRegression()
    model.fit(X, y)

    return model, X.columns

model, feature_columns = train_model(df)

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

    input_data = {
        "SUBDIVISION": subdivision,
        **monthly_values
    }

    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    if st.button("🔮 Predict Annual Rainfall"):
        prediction = model.predict(input_encoded)[0]
        st.success(f"🌧 Predicted Annual Rainfall: **{prediction:.2f} mm**")

# ================= TAB 2 =================
with tab2:
    st.subheader("📊 Monthly Rainfall Insights")

    colA, colB = st.columns(2)

    with colA:
        sub = st.selectbox("Select Subdivision", subdivisions, key="sub")

    with colB:
        month = st.selectbox("Select Month", months, key="month")

    filtered = df[df["SUBDIVISION"] == sub]

    st.metric("Average Rainfall (mm)", f"{filtered[month].mean():.2f}")
    st.metric("Minimum Recorded (mm)", f"{filtered[month].min():.2f}")
    st.metric("Maximum Recorded (mm)", f"{filtered[month].max():.2f}")

    st.markdown("### 📈 Monthly Rainfall Trend")
    st.line_chart(filtered[months])
