\# ⚡ Hourly Energy Consumption Forecast



A professional time-series forecasting project for analyzing and forecasting hourly electricity consumption using PJM power-consumption data.



The project covers the complete Data Science workflow:



\*\*Data Quality → Exploratory Data Analysis → Feature Engineering → Model Training → Time-Series Evaluation → 30-Day Forecasting → Interactive Dashboard\*\*



\---



\## 📌 Project Overview



This project forecasts hourly electricity consumption using historical PJM energy-demand data measured in megawatts (MW).



The project was developed to study:



\- Hourly electricity-consumption patterns

\- Weekday and weekend differences

\- Seasonal and long-term trends

\- Holiday-related demand patterns

\- Short-term and weekly demand dependencies

\- Machine-learning approaches for time-series forecasting

\- Future electricity demand over a 30-day horizon



The final system provides an interactive Streamlit dashboard for exploring historical consumption, model performance, and future demand forecasts.



\---



\## 🎯 Objectives



The main objectives are to:



1\. Analyze historical hourly electricity consumption.

2\. Perform data-quality checks and identify anomalies.

3\. Study hourly, weekly, seasonal, and holiday patterns.

4\. Engineer time-series features using calendar variables, lag values, and rolling statistics.

5\. Compare multiple forecasting approaches.

6\. Evaluate models using a chronological last-year holdout.

7\. Generate a recursive 30-day / 720-hour forecast.

8\. Present the results through an interactive dashboard.



\---



\## 📊 Dataset



The project uses PJM hourly power-consumption data.



\### Dataset characteristics



\- \*\*Records:\*\* 143,206 hourly observations

\- \*\*Target variable:\*\* `PJMW\_MW`

\- \*\*Timestamp:\*\* `Datetime`

\- \*\*Measurement:\*\* Electricity consumption in MW

\- \*\*Time period:\*\* April 2002 – August 2018



The source project requirement specifies that the \*\*last year of observations must be held out as the test set\*\*.



\---



\## 🧹 Data Quality \& Preprocessing



The dataset was systematically inspected before modeling.



The preprocessing workflow includes:



\- Datetime conversion

\- Chronological sorting

\- Missing-value inspection

\- Duplicate-row detection

\- Duplicate-timestamp analysis

\- Hourly interval consistency checks

\- Anomaly identification

\- Anomaly flagging

\- Conservative anomaly imputation



One extreme consumption value was identified around:



`2003-05-29 00:00:00`



The anomalous value was replaced using the mean of its neighboring observations, while retaining flags indicating that the observation was anomalous and imputed.



The processed dataset is stored in:



```text

data/processed/pjm\_energy\_cleaned.csv

