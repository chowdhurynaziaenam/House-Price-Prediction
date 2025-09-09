# House-Price-Prediction
![Python](https://img.shields.io/badge/Python-3.10-blue.svg?logo=python&logoColor=white)  
![Colab](https://img.shields.io/badge/Google-Colab-orange?logo=google-colab&logoColor=white)  
![Pandas](https://img.shields.io/badge/pandas-Data%20Analysis-blue?logo=pandas)  
![NumPy](https://img.shields.io/badge/numpy-Numerical-green?logo=numpy)  
![Matplotlib](https://img.shields.io/badge/matplotlib-Visualization-yellow)  
![Seaborn](https://img.shields.io/badge/seaborn-Stats%20Plots-lightblue)  
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)  
![Machine Learning](https://img.shields.io/badge/Models-9%20Classifiers-success)  

## 📑 Table of Contents
- [General Overview](#general-overview)
- [Project Structure](#project-structure)
- [Environment](#environment)
- [Libraries & Packages](#libraries--packages)
- [ML Models Used](#ml-models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Setup](#setup)
- [Acknowledgements](#acknowledgements)

---

## 📝 General Overview  
This house price prediction project aims to predict house prices using a machine learning model. To tackle this issue, a machine learning model trained on the House Price Prediction Dataset. A dataset containing various features of houses, such as square footage, number of bedrooms, and location, is used here to train a regression model. The model will then be able to estimate the price of a house based on its characteristics and gain valuable insights from it. The dataset contains 13 features such as- Id(to count records), MSSubClass (identifies the type of dwelling involved in the sale), MSZoning(indicates the general zoning classification of the sale), LotArea(lot size in square feet), LotConfig(Configuration of the lot), BldgType(Type of building),OverallCond (indicates the overall condition of the house), YearBuilt(denotes original construction year), YearRemodAdd(denotes remodel date & same as construction date if no remodeling or additions are done), Exterior1st(exterior covering on house), BsmtFinSF2(type 2 finished square feet), TotalBsmtSF(Total square feet of basement area), and the target variable SalePrice that needs to be predicted. 

---

## 📂 Project Structure 

The project is structured into following files:

* 'HousePricePrediction.xlsx': The dataset file in Excel format.
* 'House-Price-Prediction.ipynb': The main Python script for data processing and model training.
* 'README.md': This project documentation file.
* '.gitignore': Specifies files to be ignored by Git (e.g., temporary files, sensitive data).
* 'requirements.txt': Lists all the necessary Python libraries for the project.
* 'submission.csv': The output file containing the model's predictions.

The project included the following tasks:
- Exploratory Data Analysis (EDA)  
- Early 80:20 train-test split separating training and validation sets to avoid data leakage (for training the model-80% of the data, for evaluating the trained model-20% of the data).
- Data Cleaning (handling and imputation of missing values, checking skewness, detecting outliers using IQR and applying winsorization technique for capping extreme values)
- Applying One-hot encoder to transform categorical variable
- Developing predictive models using 3 regressor models 
- Evaluating models with multiple metrics and finding best model
- Predicting SalePrice for the test sets.

---

## 💻 Environment
- Executed using **Google Colab**.  

---

## 📚 Libraries & Packages
- 🐼 **pandas** → data manipulation  
- 🔢 **NumPy** → numerical operations  
- 📈 **Matplotlib** → plotting graphs  
- 📊 **Seaborn** → statistical visualization  
- 🤖 **Scikit-learn** → ML model development & evaluation  

---

## 🤖 ML Models Used
The following three regression models were implemented:
- Linear Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor 


---

## 📏 Evaluation Metrics
Models were evaluated using:
- Mean absolute error 
- R² value
- Root Mean Square Error (RMSE)

---

## 📊 Results
- RF and GB has the least mean absolute error of value 0.14 compared to Linear Regression with value of 0.20.
- Gradient Boosting has an R² of 0.82, and Random Forest has an R² of 0.79, compared to Linear Regression's 0.65.
- Gradient Boosting has an RMSE of approximately 37588 dollar and Random Forest has an RMSE of 39690 dollar, while Linear Regression's error is much higher at 51756 dollar.

⭐ In terms of 3 metrics, GB performs best compared to other two showing highest predictive power and the lowest average error. 


---

## ⚙️ Setup
1. Open in **Google Colab**.  
2. Upload the notebook or open it directly from Google Drive.  
3. Ensure all required packages are installed (`pip install -r requirements.txt`).  
4. Run cells sequentially to execute the analysis.  

---

## 🙏 Acknowledgements
- Original dataset [ HousePricePrediction dataset] (https://www.geeksforgeeks.org/machine-learning/house-price-prediction-using-machine-learning-in-python/)


