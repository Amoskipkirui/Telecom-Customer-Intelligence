# Telecom Customer Intelligence & Churn Prevention System

A machine learning-powered telecom analytics application designed to identify customers at risk of leaving, analyse revenue exposure, and support customer retention decisions.

This project combines data analytics, machine learning, Flask API development, and an interactive management dashboard to demonstrate how telecommunications businesses can use customer data to support retention strategies.

> **Project type:** Portfolio prototype  
> **Industry:** Telecommunications  
> **Data:** Synthetic customer and revenue data  
> **Currency:** Qatari Riyal (QAR)

---

## Project Overview

The Telecom Customer Intelligence & Churn Prevention System helps telecom managers identify customers who may leave, understand their potential revenue impact, prioritise retention activities, and track customer follow-ups.

The application includes a machine learning model, Flask API, interactive dashboard, revenue exposure analysis, and SQLite-based retention action tracker.

## Business Problem

Telecommunications companies face customer churn when subscribers discontinue their services. Losing customers can reduce recurring revenue and increase customer acquisition costs.

This project demonstrates how customer analytics and machine learning can help businesses identify potential churn, prioritise retention efforts, and organise customer follow-up activities.

## Key Features

- **Customer Intelligence Dashboard:** Displays customer statistics, churn metrics, risk distribution, and revenue exposure.
- **Customer Lookup:** Retrieves individual customer information.
- **Churn Prediction:** Estimates the probability of customer churn using a trained machine learning model.
- **Revenue Exposure Analysis:** Estimates monthly revenue exposure associated with customer churn risk.
- **Retention Priority List:** Ranks customers to support retention decisions.
- **Retention Action Tracker:** Records customer interactions, notes, and action statuses using SQLite.
- **Flask REST API:** Makes customer information, predictions, revenue analytics, and retention records accessible to the dashboard.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Data processing and application logic |
| Pandas | Data manipulation |
| Scikit-learn | Machine learning |
| Flask | API and web application |
| SQLite | Retention action storage |
| HTML and CSS | Dashboard interface |
| JavaScript | Dashboard interactions |
| Joblib | Saved model loading |
| CSV | Customer and revenue data |

## Business Value

The system demonstrates how a telecommunications business could use data to:

- Identify customers who may need retention support.
- Prioritise follow-ups using churn risk and revenue information.
- Provide actionable customer retention recommendations.
- Track customer interactions and follow-up outcomes.
- Support management decisions using customer and revenue analytics.

## Model Limitations

The selected tuned Decision Tree model achieved the following reported test results:

| Metric | Result |
|---|---:|
| Accuracy | 50.62% |
| Churn precision | 19% |
| Churn recall | 67% |
| Churn F1-score | 30% |

These results indicate that the model identifies some churn cases but also generates many false positives. Predictions should be treated as decision-support estimates rather than definitive indications that a customer will leave.

## Data and Responsible Use

This is an independent portfolio project using synthetic customer and revenue data. It is not affiliated with any specific telecommunications operator and does not use a company's internal customer records.

The system is a portfolio prototype. Additional security, authentication, testing, monitoring, and deployment work would be required before production use.

## Author

**Amos Kipkirui**

Data Analytics | Machine Learning | Flask API Development
