import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page config
st.set_page_config(page_title="Sales Prediction", layout="wide")

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="title-card">
    <h1>📊 Sales Prediction using ML</h1>
    <p>Simple & Multiple Linear Regression</p>
</div>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("advertising.csv")

df = load_data()

# Sidebar prediction
st.sidebar.header("Real-time Prediction")
st.sidebar.write("Adjust advertising budget:")

tv = st.sidebar.slider("TV Budget", float(df.TV.min()), float(df.TV.max()), 100.0)
radio = st.sidebar.slider("Radio Budget", float(df.Radio.min()), float(df.Radio.max()), 20.0)
news = st.sidebar.slider("Newspaper Budget", float(df.Newspaper.min()), float(df.Newspaper.max()), 30.0)

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# =========================
# SIMPLE LINEAR REGRESSION
# =========================
X_s = df[["TV"]]
y = df["Sales"]

Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(
    X_s, y, test_size=0.2, random_state=42
)

sc_s = StandardScaler()
Xtr_s_scaled = sc_s.fit_transform(Xtr_s)
Xte_s_scaled = sc_s.transform(Xte_s)

slr = LinearRegression().fit(Xtr_s_scaled, ytr_s)
y_pred_s = slr.predict(Xte_s_scaled)

# =========================
# MULTIPLE LINEAR REGRESSION
# =========================
features = ["TV", "Radio", "Newspaper"]
X_m = df[features]

Xtr_m, Xte_m, ytr_m, yte_m = train_test_split(
    X_m, y, test_size=0.2, random_state=42
)

sc_m = StandardScaler()
Xtr_m_scaled = sc_m.fit_transform(Xtr_m)
Xte_m_scaled = sc_m.transform(Xte_m)

mlr = LinearRegression().fit(Xtr_m_scaled, ytr_m)
y_pred_m = mlr.predict(Xte_m_scaled)

# Sidebar prediction output
user_scaled = sc_m.transform([[tv, radio, news]])
pred_sales = mlr.predict(user_scaled)[0]

st.sidebar.divider()
st.sidebar.subheader("📈 Predicted Sales")
st.sidebar.success(f"{pred_sales:.2f} units")

# =========================
# VISUALIZATION
# =========================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 SLR: TV Budget vs Sales")

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mean_absolute_error(yte_s, y_pred_s):.2f}")
    m2.metric("RMSE", f"{np.sqrt(mean_squared_error(yte_s, y_pred_s)):.2f}")
    m3.metric("R²", f"{r2_score(yte_s, y_pred_s):.2f}")

    fig, ax = plt.subplots()
    ax.scatter(df.TV, df.Sales, alpha=0.4)

    x_line = pd.DataFrame(
        np.linspace(df.TV.min(), df.TV.max(), 100),
        columns=["TV"]
    )
    ax.plot(
        x_line,
        slr.predict(sc_s.transform(x_line)),
        color="red",
        linewidth=3
    )

    ax.set_xlabel("TV Budget")
    ax.set_ylabel("Sales")
    st.pyplot(fig)

with col2:
    st.markdown("### 📉 MLR: Feature Correlation")

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mean_absolute_error(yte_m, y_pred_m):.2f}")
    m2.metric("RMSE", f"{np.sqrt(mean_squared_error(yte_m, y_pred_m)):.2f}")
    m3.metric("R²", f"{r2_score(yte_m, y_pred_m):.2f}")

    fig, ax = plt.subplots()
    sns.heatmap(df[features].corr(), annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig)

# =========================
# ACTUAL vs PREDICTED
# =========================
st.markdown("### 🎯 Actual vs Predicted Sales (MLR)")

fig, ax = plt.subplots()
ax.scatter(yte_m, y_pred_m, alpha=0.6)
ax.plot(
    [yte_m.min(), yte_m.max()],
    [yte_m.min(), yte_m.max()],
    linestyle="--"
)
ax.set_xlabel("Actual Sales")
ax.set_ylabel("Predicted Sales")
st.pyplot(fig)

st.success("✅ App executed successfully")
