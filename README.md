# 📊 Advertising Sales Prediction (ML)

A machine learning web application that predicts **product sales** based on **advertising spend** across **TV, Radio, and Newspaper** using **Linear Regression**, built with **Python and Streamlit**.

---

## 🚀 Features

* 📈 **Simple Linear Regression** (TV → Sales)
* 📉 **Multiple Linear Regression** (TV, Radio, Newspaper → Sales)
* 🎯 **Real-time Sales Prediction** using sidebar sliders
* 📊 **Model Evaluation Metrics** (MAE, RMSE, R²)
* 🔥 **Correlation Heatmap**
* 🎨 **Modern UI** with gradient background and styled metrics
* 🖥️ Interactive **Streamlit Dashboard**

---

## 🧠 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit

---

## 📁 Project Structure

```
ad-sales-ml/
│
├── app.py
├── advertising.csv
├── style.css
├── requirements.txt
├── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install requirements.txt
```

### 2️⃣ Run the App

```bash
streamlit run app.py
```

---

## 📊 Dataset

The dataset contains advertising budgets and corresponding sales:

| Feature   | Description                     |
| --------- | ------------------------------- |
| TV        | TV advertising budget           |
| Radio     | Radio advertising budget        |
| Newspaper | Newspaper advertising budget    |
| Sales     | Product sales (target variable) |

---

## 🎯 Output

* Predicts **sales value** for given advertising budgets
* Visualizes regression trends
* Compares **actual vs predicted sales**
* Helps understand **impact of ads on sales**

---

## 🎓 Use Cases

* Marketing analytics
* Sales forecasting
* Data science mini/major project
* Resume & portfolio project

---

## ✨ Future Enhancements

* Add confidence intervals
* Model comparison (Ridge, Lasso)
* Export prediction reports
* Deploy on Streamlit Cloud
