# 🏠 House Price Prediction System

A professional Machine Learning web application that predicts residential house prices using a trained **Support Vector Regression (SVR)** model. The application provides both **single-house prediction** through an interactive form and **batch prediction** using CSV uploads.

---

## 🚀 Live Demo

**Live Website:**
`[https://your-render-url.onrender.com](https://house-price-prediction-website-34cm.onrender.com/)`

---

## 📌 Project Overview

The House Price Prediction System estimates the selling price of a residential property based on its characteristics. It is built using the **Ames Housing Dataset** and deployed as a Flask web application.

Users can:

* Predict the price of a single house using a web form.
* Upload a CSV file to predict prices for multiple houses simultaneously.
* Download the prediction results.

---

## ✨ Features

* 🏠 Single House Price Prediction
* 📄 Batch Prediction using CSV Upload
* ⚡ Machine Learning Powered (SVR)
* 🔧 Automatic Feature Engineering
* 📊 Data Preprocessing Pipeline
* 📱 Responsive Bootstrap 5 Interface
* 💬 Flash Messages
* ⏳ Loading Spinner
* ✅ Input Validation
* 📥 Download Batch Prediction Results

---

## 🧠 Machine Learning Model

| Item          | Description                     |
| ------------- | ------------------------------- |
| Algorithm     | Support Vector Regression (SVR) |
| Dataset       | Ames Housing Dataset            |
| Features Used | 65 Features                     |
| Framework     | Scikit-learn                    |

---

## ⚙️ Feature Engineering

The application automatically creates engineered features before prediction:

* HouseAge
* RemodelAge
* GarageAge
* TotalBathrooms
* TotalSF
* TotalPorchArea
* HasGarage
* HasBasement
* HasFireplace

---

## 🛠 Tech Stack

### Frontend

* HTML5
* CSS3
* Bootstrap 5
* Jinja2
* Bootstrap Icons

### Backend

* Python
* Flask

### Machine Learning

* Scikit-learn
* Pandas
* NumPy
* Joblib

### Deployment

* Render
* Gunicorn

---

## 📂 Project Structure

```text
house-price/
│
├── app.py
├── config.py
├── utils.py
├── feature_config.py
├── requirements.txt
├── runtime.txt
│
├── model/
│   └── house_price_svr_pipeline.pkl
│
├── templates/
│   ├── base.html
│   ├── index.html
│   └── about.html
│
├── static/
│   ├── css/
│   └── images/
│
└── uploads/
```

---

## 🔄 Prediction Workflow

```text
User Input
      │
      ▼
Validation
      │
      ▼
Feature Engineering
      │
      ▼
Preprocessing Pipeline
      │
      ▼
Support Vector Regression Model
      │
      ▼
Predicted House Price
```

---

## 📄 Batch Prediction

The application supports batch prediction.

### Steps

1. Prepare a CSV file containing all required input features.
2. Upload the file.
3. The model predicts prices for all records.
4. Download the resulting CSV.

---

## 💻 Installation

Clone the repository

```bash
git clone https://github.com/Anwar6777/house-price-prediction.git
```

Navigate into the project

```bash
cd house-price-prediction
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

---

## 🌐 Deployment (Render)

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
gunicorn app:app
```

---

## 📊 Dataset

* Ames Housing Dataset

---

## ⚠️ Disclaimer

The predicted values are generated using historical housing data and machine learning techniques.

These predictions should be considered estimates and **not official property valuations**.

---

## 👨‍💻 Developer

**Anwar Ansari**

MCA Student
Machine Learning & AI Enthusiast

GitHub: [https://github.com/Anwar6777](https://github.com/Anwar6777)

LinkedIn: [https://linkedin.com/in/anwar-ansari-945114b338](https://linkedin.com/in/anwar-ansari-945114b338)

Email: [anwaransari66763@gmail.com](mailto:anwaransari66763@gmail.com)

---

## ⭐ Future Improvements

* CatBoost/XGBoost Model Integration
* Model Comparison Dashboard
* Interactive Data Visualizations
* User Authentication
* Prediction History
* API Support
* Docker Deployment

---

## 📜 License

This project is developed for educational and portfolio purposes. All rights reserved.
