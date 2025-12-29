import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Sales Prediction using ML", layout="wide")

# --------------------------------------------------
# LOAD CSS (SAFE)
# --------------------------------------------------
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown("""
<div class="title-card">
    <h1>📊 Sales Prediction using ML</h1>
    <p>Simple & Multiple Linear Regression</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA (DEPLOYMENT SAFE)
# --------------------------------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "advertising.csv")
    return pd.read_csv(file_path)

df = load_data()

# --------------------------------------------------
# SIDEBAR – REAL-TIME PREDICTION
# --------------------------------------------------
st.sidebar.header("🚀 Real-time Prediction")
st.sidebar.write("Adjust advertising budget:")

tv = st.sidebar.slider(
    "TV Budget",
    float(df["TV"].min()),
    float(df["TV"].max()),
    float(df["TV"].mean())
)

radio = st.sidebar.slider(
    "Radio Budget",
    float(df["Radio"].min()),
    float(df["Radio"].max()),
    float(df["Radio"].mean())
)

news = st.sidebar.slider(
    "Newspaper Budget",
    float(df["Newspaper"].min()),
    float(df["Newspaper"].max()),
    float(df["Newspaper"].mean())
)

# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# --------------------------------------------------
# SIMPLE LINEAR REGRESSION (TV → Sales)
# --------------------------------------------------
X_slr = df[["TV"]]
y = df["Sales"]

Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(
    X_slr, y, test_size=0.2, random_state=42
)

sc_slr = StandardScaler()
Xtr_s_scaled = sc_slr.fit_transform(Xtr_s)
Xte_s_scaled = sc_slr.transform(Xte_s)

slr_model = LinearRegression()
slr_model.fit(Xtr_s_scaled, ytr_s)
y_pred_s = slr_model.predict(Xte_s_scaled)

# --------------------------------------------------
# MULTIPLE LINEAR REGRESSION (TV + Radio + Newspaper)
# --------------------------------------------------
features = ["TV", "Radio", "Newspaper"]
X_mlr = df[features]

Xtr_m, Xte_m, ytr_m, yte_m = train_test_split(
    X_mlr, y, test_size=0.2, random_state=42
)

sc_mlr = StandardScaler()
Xtr_m_scaled = sc_mlr.fit_transform(Xtr_m)
Xte_m_scaled = sc_mlr.transform(Xte_m)

mlr_model = LinearRegression()
mlr_model.fit(Xtr_m_scaled, ytr_m)
y_pred_m = mlr_model.predict(Xte_m_scaled)

# --------------------------------------------------
# SIDEBAR OUTPUT
# --------------------------------------------------
user_input = pd.DataFrame([[tv, radio, news]], columns=features)
user_scaled = sc_mlr.transform(user_input)
predicted_sales = mlr_model.predict(user_scaled)[0]

st.sidebar.divider()
st.sidebar.subheader("📈 Predicted Sales")
st.sidebar.success(f"{predicted_sales:.2f} units")

# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 SLR: TV Budget vs Sales")

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mean_absolute_error(yte_s, y_pred_s):.2f}")
    m2.metric("RMSE", f"{np.sqrt(mean_squared_error(yte_s, y_pred_s)):.2f}")
    m3.metric("R²", f"{r2_score(yte_s, y_pred_s):.2f}")

    fig1, ax1 = plt.subplots()
    ax1.scatter(df["TV"], df["Sales"], alpha=0.4)

    x_line = pd.DataFrame(
        np.linspace(df["TV"].min(), df["TV"].max(), 100),
        columns=["TV"]
    )
    ax1.plot(
        x_line,
        slr_model.predict(sc_slr.transform(x_line)),
        color="red",
        linewidth=3
    )

    ax1.set_xlabel("TV Budget")
    ax1.set_ylabel("Sales")
    st.pyplot(fig1)

with col2:
    st.markdown("### 📉 MLR: Feature Correlation")

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mean_absolute_error(yte_m, y_pred_m):.2f}")
    m2.metric("RMSE", f"{np.sqrt(mean_squared_error(yte_m, y_pred_m)):.2f}")
    m3.metric("R²", f"{r2_score(yte_m, y_pred_m):.2f}")

    fig2, ax2 = plt.subplots()
    sns.heatmap(df[features].corr(), annot=True, cmap="Blues", ax=ax2)
    st.pyplot(fig2)

# --------------------------------------------------
# ACTUAL vs PREDICTED
# --------------------------------------------------
st.markdown("### 🎯 Actual vs Predicted Sales (MLR)")

fig3, ax3 = plt.subplots()
ax3.scatter(yte_m, y_pred_m, alpha=0.6)
ax3.plot(
    [yte_m.min(), yte_m.max()],
    [yte_m.min(), yte_m.max()],
    linestyle="--"
)
ax3.set_xlabel("Actual Sales")
ax3.set_ylabel("Predicted Sales")
st.pyplot(fig3)

st.success("✅ App executed successfully")
