# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
# Import the necessary library for Ridge regression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('housing.csv')

# Display the data types of each column
print(data.dtypes)

# Drop the columns "id" and "Unnamed: 0"
data.drop(['id', 'Unnamed: 0'], axis=1, inplace=True)

# Display a statistical summary of the data
summary = data.describe()

# Display the summary
print(summary)

# Count the number of houses with unique floor values and convert to DataFrame
floor_counts = data['floors'].value_counts().to_frame()

# Display the floor counts
#print(floor_counts)

# Create a boxplot to analyze price outliers based on waterfront view
plt.figure(figsize=(10, 6))
sns.boxplot(x='waterfront', y='price', data=data)
plt.title('Boxplot of Price by Waterfront View')
plt.xlabel('Waterfront (0 = No, 1 = Yes)')
plt.ylabel('Price')
plt.show()

# Create a regplot to determine the correlation between sqft_above and price
plt.figure(figsize=(10, 6))
sns.regplot(x='sqft_above', y='price', data=data, line_kws={"color": "red"})
plt.title('Regression Plot of sqft_above vs Price')
plt.xlabel('Square Footage Above Ground (sqft_above)')
plt.ylabel('Price')
plt.show()

# Fit a linear regression model to predict price using sqft_living
X = data[['sqft_living']]  # Feature
y = data['price']  # Target
# Initialize the linear regression model
model = LinearRegression()
# Fit the model to the data
model.fit(X, y)
# Predict the prices using the model
y_pred = model.predict(X)
# Calculate the R^2 score
r2 = r2_score(y, y_pred)
# Display the R^2 value
print(f'R^2 score: {r2}')

# Separate numeric columns from non-numeric columns for handling NaN values
numeric_data = data.select_dtypes(include=[float, int])
non_numeric_data = data.select_dtypes(exclude=[float, int])
# Handle missing values in numeric columns by replacing NaN with the mean
imputer = SimpleImputer(strategy='mean')
numeric_data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)
# Combine the imputed numeric data with the non-numeric data
data_imputed = pd.concat([numeric_data_imputed, non_numeric_data], axis=1)
# Fit a linear regression model to predict price using multiple features
features = ['floors', 'waterfront', 'lat', 'bedrooms', 'sqft_basement', 'view', 
            'bathrooms', 'sqft_living15', 'sqft_above', 'grade', 'sqft_living']
X_multi = data_imputed[features]  # Multiple features
y = data_imputed['price']  # Target
# Initialize the linear regression model
model_multi = LinearRegression()
# Fit the model to the data
model_multi.fit(X_multi, y)
# Predict the prices using the model
y_pred_multi = model_multi.predict(X_multi)
# Calculate the R^2 score for the multi-feature model
r2_multi = r2_score(y, y_pred_multi)
# Display the R^2 value for the multi-feature model
print(f'R^2 score for multiple features: {r2_multi}')

# Create a pipeline object that scales the data, performs a polynomial transform, and fits a linear regression model
pipeline = Pipeline([
    ('scaler', StandardScaler()),                   # Scaling the data
    ('poly', PolynomialFeatures(degree=2)),         # Polynomial transformation (degree=2)
    ('linear_regression', LinearRegression())       # Linear regression model
])
# Fit the pipeline to the data
pipeline.fit(X_multi, y)
# Predict the prices using the model
y_pred_pipeline = pipeline.predict(X_multi)
# Calculate the R^2 score for the pipeline model
r2_pipeline = r2_score(y, y_pred_pipeline)
# Display the R^2 value for the pipeline model
print(f'R^2 score for multiple features with polynomial transform: {r2_pipeline}')

# Define the features and target variable
features = ['floors', 'waterfront', 'lat', 'bedrooms', 'sqft_basement', 'view', 
            'bathrooms', 'sqft_living15', 'sqft_above', 'grade', 'sqft_living']
X = data[features]
y = data['price']
# Handle missing values in numeric columns by replacing NaN with the mean using an imputer
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
# Create a Ridge regression model with regularization parameter alpha=0.1
ridge_model = Ridge(alpha=0.1)
# Fit the model to the training data
ridge_model.fit(X_train, y_train)
# Predict the prices using the test data
y_pred_ridge = ridge_model.predict(X_test)
# Calculate the R² score for Ridge regression using the test data
r2_ridge = r2_score(y_test, y_pred_ridge)
# Display the R² score
print(f'R² score for Ridge regression: {r2_ridge}')

# Perform a second-order polynomial transform on both the training data and testing data
poly = PolynomialFeatures(degree=2)
# Transform the training and testing data
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
# Create a Ridge regression model with regularization parameter alpha=0.1
ridge_model_poly = Ridge(alpha=0.1)
# Fit the model to the polynomial transformed training data
ridge_model_poly.fit(X_train_poly, y_train)
# Predict the prices using the polynomial transformed test data
y_pred_ridge_poly = ridge_model_poly.predict(X_test_poly)
# Calculate the R² score for Ridge regression with polynomial features using the test data
r2_ridge_poly = r2_score(y_test, y_pred_ridge_poly)
# Display the R² score for Ridge regression with polynomial features
print(f'R² score for Ridge regression with polynomial transform: {r2_ridge_poly}')
