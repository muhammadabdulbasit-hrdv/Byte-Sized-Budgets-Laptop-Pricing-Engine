# Byte-Sized-Budgets-Laptop-Pricing-Engine
End-to-end ML pipeline predicting laptop prices using a Gradient Boosting Regressor on 1,275 hardware profiles. Handles a right-skewed market and complex feature interactions. Insights reveal RAM (0.40) and SSD (0.13) as core price drivers, cleanly isolating premium configurations.

# Laptop Price Prediction Engine

An end-to-end machine learning pipeline that decodes market dynamics and predicts laptop pricing structures using a **Gradient Boosting Regressor**. 

## 📌 Project Overview
This project analyzes 1,275 unique hardware configurations across 15 distinct specifications to understand how granular hardware ecosystems, brand equity, and form factors dictate final retail valuations. 

## 📊 Key Insights & EDA
* **Target Variable:** The `Price (Euro)` distribution is heavily right-skewed, featuring premium outliers peaking between 4,000 and 6,000 Euros.
* **Market Dominance:** High-volume trends are driven by Windows 10 Notebooks from HP, Dell, and Lenovo.
* **Price Volatility:** Premium pricing and extreme variance are heavily driven by luxury brands (Razer) and specialized form factors (Workstations).

## 🤖 Modeling & Feature Importance
A **Gradient Boosting Regressor** was selected for its robust handling of non-linear target distributions and outlier boundaries. Hardware tiers serve as the primary structural anchors for pricing predictions:

| Feature | Importance Score | Market Reality |
| :--- | :--- | :--- |
| **RAM (GB)** | **0.40** | Primary anchor segmenting budget and premium units |
| **SSD Storage (GB)** | **0.13** | Reflects exponential pricing scaling for modern storage |
| **TypeName_Notebook** | **0.09** | Captures commodity high-volume vs. low-volume luxury |
| **Weight (kg)** | **0.087** | Form factor proxy (dense ultrabooks vs. heavy gaming rigs) |
| **Intel Core i7** | **0.057** | Threshold split flag for premium tier performance |
| **Nvidia Quadro** | **0.042** | Maps hyper-expensive, specialized workstation configurations |

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib



* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib
