# AI Jobs & Salary Project

Data Mining project analyzing the impact of Artificial Intelligence on job survival and salary trends.

---

## Team Members

* Mariam Khalil
* Samira Gamal

---

## Project Overview

Artificial Intelligence is reshaping the global job market. Some roles are becoming obsolete, while others are growing rapidly due to new technological demands.

This project focuses on analyzing how AI affects:

* Job survival probability
* Salary trends across roles and countries
* Skill demand and its impact on job stability

The project also builds predictive models and an interactive dashboard to support decision-making.

---

## Objectives

* Analyze the impact of AI risk on job survival
* Identify high-risk and low-risk job categories
* Study salary variations across countries and experience levels
* Predict job survival class using classification models
* Predict salary using regression models
* Build an interactive dashboard for data exploration and insights

---

## Dataset

**Name:** Future of Jobs AI Dataset (2015–2035)
**Source:** Kaggle

---

## Dataset Description

* Total Records: 12,343
* Time Range: 2015 to 2035
* Countries: USA, UK, Canada, India, Germany, Australia
* Job Titles: 10 different roles

---

## Features

* `job_title`
* `country`
* `experience_level`
* `education_level`
* `year`
* `salary`
* `ai_risk_score`
* `primary_skill`
* `skill_demand_score`
* `job_openings`

---

## Target Variables

* `job_survival_class`

  * 0 → Low survival
  * 1 → Medium survival
  * 2 → High survival

* `salary`

  * Continuous variable used for regression

---

## Data Quality Notes

* Data after 2025 includes projected values
* Salary distribution is imbalanced toward high salaries
* No missing values detected
* Categorical features require encoding
* Numerical features may require normalization

---

## Data Mining Approach (CRISP-DM)

### 1. Business Understanding

Define the problem of AI impact on jobs and salaries.
Set objectives for prediction and analysis.

### 2. Data Understanding

Explore distributions, correlations, and trends.
Identify patterns and anomalies.

### 3. Data Preparation

* Encode categorical variables
* Normalize numerical features
* Split data into training and testing sets

### 4. Modeling

#### Classification (Job Survival)

* Logistic Regression
* Random Forest Classifier

#### Regression (Salary Prediction)

* Linear Regression
* Random Forest Regressor

### 5. Evaluation

* Classification: Accuracy, F1-score, Confusion Matrix
* Regression: RMSE, MAE

### 6. Deployment

Interactive Streamlit dashboard for visualization and insights.

---

## Dashboard Features

* Global filters (year, country, job title, etc.)
* KPI summary cards
* Data exploration charts
* Salary analysis section
* Job survival analysis section
* Business insights panel
* Phase 2 roadmap

---

## Project Structure

```
dm-ai-jobs-dashboard/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
├── notebooks/
├── src/
├── assets/
└── docs/
```

---

## How to Run Locally

### 1. Install dependencies

pip install -r requirements.txt

### 2. Run the dashboard

streamlit run app.py

---

## Phase 1 Status

* Problem defined
* Dataset selected and analyzed
* CRISP-DM plan prepared
* Initial dashboard prototype created
* Exploratory data analysis completed

---

## Phase 2 Plan

* Data preprocessing and feature engineering
* Train and tune machine learning models
* Evaluate model performance
* Integrate model results into dashboard
* Prepare final presentation

---

## Future Work

* Improve model accuracy
* Add more advanced visualizations
* Deploy dashboard online
* Expand dataset with more countries or roles

---

## Tools & Technologies

* Python
* Pandas
* NumPy
* Plotly
* Streamlit
* Scikit-learn
* Git & GitHub
