
# README

## Overview

This project involves predictive modeling using housing data to estimate property prices based on several features such as square footage, number of floors, and waterfront views. The models used include Linear Regression, Ridge Regression, and Polynomial Regression, with an additional Ridge model utilizing polynomial transformations. Data pre-processing includes handling missing values and scaling for better performance.

## Dataset

The dataset used for this project is `housing.csv`, which contains various attributes related to houses, such as:
- Square footage
- Number of bedrooms and bathrooms
- Waterfront view
- Location (latitude)
- Grade
- Price (target variable)

## Requirements

- Python 3.x
- Required Libraries:
  - pandas
  - seaborn
  - matplotlib
  - scikit-learn
  
You can install the required libraries by running:

```bash
pip install pandas seaborn matplotlib scikit-learn
```

## Features

1. **Data Cleaning:**
   - Dropped irrelevant columns (`id`, `Unnamed: 0`).
   - Handled missing values in numeric columns by replacing them with the mean.

2. **Data Visualization:**
   - Created boxplots to analyze outliers based on waterfront views.
   - Used regression plots to examine correlations between square footage above ground and price.

3. **Linear Regression Models:**
   - **Single Feature Regression:** Predicted price using `sqft_living` and calculated the R² score.
   - **Multi-Feature Regression:** Used several features to improve the prediction and calculated the corresponding R² score.

4. **Pipeline with Polynomial Regression:**
   - Created a pipeline that scales the data, performs a polynomial transformation (degree 2), and fits a linear regression model.
   - Displayed the R² score for the polynomial regression model.

5. **Ridge Regression:**
   - Implemented Ridge Regression with a regularization parameter (`alpha=0.1`).
   - Split the data into training and testing sets (80% training, 20% testing).
   - Computed the R² score for Ridge regression on the test set.
   - Applied polynomial features to Ridge Regression for improved performance.

## How to Run

1. Clone the repository and ensure the `housing.csv` file is in the same directory as the script.
2. Run the Python script to clean the data, fit regression models, and display results:

```bash
python housing_price_prediction.py
```

3. The script will output the R² scores for the various models:
   - Single feature Linear Regression.
   - Multi-feature Linear Regression.
   - Ridge Regression.
   - Ridge Regression with Polynomial Features.

## Results

- **Linear Regression:**
  - Single Feature (sqft_living) R² score.
  - Multi-feature R² score.

- **Ridge Regression:**
  - R² score for Ridge Regression with `alpha=0.1`.

- **Polynomial Ridge Regression:**
  - R² score for Ridge Regression with second-degree polynomial features.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
