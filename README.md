# Byte-Sized-Budgets-Laptop-Pricing-Engine
End-to-end ML pipeline predicting laptop prices using a Gradient Boosting Regressor on 1,275 hardware profiles. Handles a right-skewed market and complex feature interactions. Insights reveal RAM (0.40) and SSD (0.13) as core price drivers, cleanly isolating premium configurations.

# 💻 Laptop Price Prediction Engine

An end-to-end machine learning pipeline that decodes market dynamics and predicts laptop pricing structures using a **Gradient Boosting Regressor**. This project processes mixed hardware configurations, handles structural market anomalies, and maps complex feature interactions to deliver highly generalized price valuations.

---

## 📌 Project Overview
The dataset consists of **1,275 rows and 15 columns** with zero missing (null) values and no duplicate rows, ensuring high data integrity from the start.

### 🔍 Core EDA Operations (Exploratory Data Analysis)
* **Target Distribution Analysis:** Evaluated `Price (Euro)` and identified a heavy **right-skewed distribution**, uncovering a distinct tail of premium/luxury outliers ranging between 4,000 to 6,000 Euros.
* **Univariate Trends & Market Share:** Analyzed volume distribution across brands and form factors. Found that HP, Dell, and Lenovo dominate volume, Windows 10 is the overwhelmingly popular OS, and generic Notebooks represent the bulk of the market.
* **Anomaly & Niche Spotting:** Isolated low-sample segments like Samsung (only 9 units total) and highlighted price volatility in niche sectors like Razer (luxury laptops) and Windows 7/10 S environments.
* **Bivariate Price Drivers:** Evaluated categorical features against average pricing using bar plots to isolate premium tiers. Discovered that Intel dominates CPU volume while Nvidia leads in high-end specialized GPU configurations.
* **Correlation Matrix Analysis:** Identified strong linear correlations between price and internal hardware (RAM, CPU Frequency), while physical screen size (`Inches`) showed a surprisingly weak correlation.

---

## 🛠️ Feature Engineering Pipeline
Before feeding the data into the model, raw technical specifications and text strings were transformed into clean, machine-learning-ready numeric nodes:

1. **Text Stripping & Numeric Casting:** Extracted raw integers/floats from string columns by stripping units (e.g., removing `"GB"` from RAM and `"kg"` from Weight).
2. **Resolution Feature Extraction:** Parsed the complex `ScreenResolution` string column to isolate raw pixel counts, creating explicit numerical dimensions like **`X_resolution`**.
3. **Hardware Tiers & Simplification:** Grouped fragmented, high-cardinality text data into broad, meaningful categories (e.g., consolidating complex CPU descriptions into a clean **`CPU_Brand_Intel Core i7`** indicator).
4. **Specialized Category Flagging:** Isolated hyper-expensive, industrial-grade components into specific categories like **`GPU_Brand_Type_Nvidia Quadro`** to help the model identify workstation-class pricing boundaries.
5. **One-Hot Encoding:** Converted all processed categorical columns (`Company`, `TypeName`, `CPU_Brand`, etc.) into binary dummy variables (`0` or `1`) using One-Hot Encoding so the decision trees could execute clean node splits.

---

## 🤖 Modeling & Feature Importance
A **Gradient Boosting Regressor** was deployed as the final engine due to its superior ability to sequentially isolate premium outlier boundaries without getting thrown off by small sample sizes. 

The top hardware drivers prioritized by the algorithm include:

| Feature | Importance Score | Market Interpretation |
| :--- | :--- | :--- |
| **RAM (GB)** | **0.40** | Primary structural anchor; instantly splits budget units from luxury machines. |
| **SSD_GB** | **0.13** | Captures the exponential scaling of modern solid-state storage pricing. |
| **TypeName_Notebook** | **0.09** | Acts as a high-volume penalty split to separate standard commodity laptops. |
| **Weight (kg)** | **0.087** | Form factor proxy that differentiates sleek ultrabooks from dense gaming rigs. |
| **CPU_Brand_Intel Core i7** | **0.057** | Higher tier processing threshold indicator for premium performance. |
| **GPU_Brand_Type_Nvidia Quadro**| **0.042** | Maps hyper-expensive, specialized workstation configurations (4,000+ Euros). |

---

## ⚙️ Tech Stack & Toolkit
* **Language:** Python
* **Data Processing & EDA:** Pandas, NumPy, Visualizations (Seaborn, Matplotlib)
* **Machine Learning:** Scikit-Learn (Gradient Boosting Regressor, Preprocessing, Target Encoding)
