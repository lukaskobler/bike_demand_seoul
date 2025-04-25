This repository contains a machine learning project focused on predicting hourly bike rental demand in Seoul using historical weather and rental data. The goal is to build and evaluate regression models that support smarter fleet distribution, cost reduction, and planning in urban bike-sharing systems.

The project uses the public Seoul Bike Sharing Demand dataset, featuring variables such as temperature, humidity, wind speed, and visibility. The complete workflow includes data preprocessing, regression modeling (using XGBoost), performance evaluation, and both time-based and randomly sampled data splits to support model validation.

Key Features:

Structured dataset with clear metadata and regression/forecasting splits
XGBoost models with basic hyperparameter tuning and evaluation (RMSE, MAE, MAPE, R²)
Visualizations comparing actual vs. predicted demand, with support for resampling over time
Transparent codebase designed with FAIR (Findable, Accessible, Interoperable, Reproducible) principles in mind
Reproducible environment configuration using modern tooling
Environment Setup:

To initialize the environment, run:

./setup.sh
This script creates a virtual environment and installs all dependencies listed in requirements.txt. The file was generated using pip-compile (from the pip-tools package), providing a reproducible and consistent dependency resolution process that goes beyond a basic pip install.
