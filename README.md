# HR Employee Attrition Prediction System

A professional data science project that predicts employee attrition using machine learning and presents the results through an interactive Streamlit dashboard.

## Overview
This project addresses a practical business problem in human resources: identifying employees who are likely to leave the company. It demonstrates an end-to-end data science workflow, including data preprocessing, exploratory data analysis, model building, evaluation, and deployment as a web application.

## Why This Project Matters
Employee attrition affects productivity, hiring costs, and team stability. This project helps HR teams and managers identify potential employee turnover risks early and take preventive actions.

## Key Features
- Data cleaning and preprocessing
- Exploratory data analysis with visual insights
- Machine learning model training and comparison
- Evaluation using accuracy, F1-score, and ROC-AUC
- Interactive dashboard for real-time predictions

## Tech Stack
- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Streamlit
- joblib

## Project Structure
```text
HR EMPLOYEE/
├── hr_attrition_model.py
├── app_streamlit.py
├── requirements.txt
├── README.md
└── outputs/
```

## Screenshots
### EDA and Insights
![EDA Analysis](outputs/eda_plots.png)

### Model Evaluation
![Model Evaluation](outputs/model_evaluation.png)

## Dataset
The project supports any CSV file containing an Attrition column with values such as Yes/No. If no dataset is provided, it can generate a synthetic IBM-style HR dataset for demonstration purposes.

## Installation
```bash
pip install -r requirements.txt
```

## Run the Project
### 1. Run the training and analysis pipeline
```bash
python hr_attrition_model.py
```

### 2. Launch the Streamlit dashboard
```bash
streamlit run app_streamlit.py
```
Then open the local URL shown in the terminal.

## Models Used
- Logistic Regression
- Random Forest
- Gradient Boosting

## Evaluation Metrics
- Accuracy
- F1-score
- ROC-AUC
- Confusion Matrix

## Skills Demonstrated
This project highlights important skills for aspiring data scientists and analysts:
- Data preprocessing
- Exploratory data analysis
- Machine learning
- Model evaluation
- Dashboard development
- Business problem solving

## Resume-Ready Summary
Built an HR employee attrition prediction system using Python, machine learning, and Streamlit to identify employees at risk of leaving. The project includes data analysis, model comparison, and an interactive dashboard for practical decision-making.

## Future Improvements
- Deploy the app on Streamlit Cloud
- Add SHAP-based model explanations
- Use a real public HR dataset for higher accuracy
- Improve the user interface for a more polished presentation

## Author
<<<<<<< HEAD
HARIKRISHNA S
Data Science Internship Project
>>>>>>> 2ee52bc (Initial commit)
=======
Hari
Data Science Internship Project
>>>>>>> 075ac1d (Polish README and gitignore)
