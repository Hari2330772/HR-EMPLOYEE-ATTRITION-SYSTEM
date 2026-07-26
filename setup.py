from setuptools import setup, find_packages

setup(
    name="hr-attrition-prediction",
    version="1.0.0",
    description="HR employee attrition prediction project with Streamlit dashboard",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5",
        "numpy>=1.23",
        "matplotlib>=3.6",
        "seaborn>=0.12",
        "scikit-learn>=1.2",
        "joblib>=1.2",
        "streamlit>=1.25",
        "xgboost>=1.7",
    ],
)
