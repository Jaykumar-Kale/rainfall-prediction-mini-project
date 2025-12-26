# Rainfall Prediction Mini Project

This repository contains a **Rainfall Prediction System** built using **Machine Learning** and deployed as a **Streamlit web app**.  
The app uses real rainfall data to provide both *annual rainfall predictions* and *monthly rainfall insights*.

🔗 **Live Demo:**  
https://jaykumar-kale-rainfall-prediction-mini-project-app-h0bgtn.streamlit.app/

---

## 🧠 Project Overview

This application predicts **annual rainfall** for subdivisions in India using a **Linear Regression** model trained on historical rainfall data.  
It also provides **monthly historical statistics** like average, minimum, and maximum rainfall with interactive plots.

---

## 📊 App Features

### 1. Annual Rainfall Prediction
- Choose a subdivision
- Enter monthly rainfall values
- Predict annual rainfall using ML

### 2. Monthly Rainfall Insights
- Select a subdivision and month
- View historical statistics
- Trend visualization with charts

---

## 🔧 Tech Stack

- Python
- Pandas, Scikit-learn (ML)
- Streamlit (Web App)
- GitHub (Version Control)
- Streamlit Cloud (Deployment)

---

## 📁 Folder Structure

```
rainfall-prediction-mini-project/
├── app.py
├── train_model.py
├── data/
│   └── rainfall.csv
├── model/ (optional)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to Run Locally

1. Clone this repo
2. Create and activate a virtual environment
3. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run app.py
```

---

## 📝 Notes

- The model is trained **on the fly** in the Streamlit app (no saved `.pkl` files required).
- Dataset sourced from Kaggle (real historical rainfall data).
