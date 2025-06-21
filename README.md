# House Price Prediction 
This project focuses on building a machine learning model to predict house prices using a Random Forest Regressor. 
The dataset used for this project is the California Housing Dataset, which includes various factors influencing housing prices, such as location (longitude and latitude), median income, total rooms, total bedrooms, population, and housing median age.

The goal of this project is to create a robust predictive model that provides accurate price estimates. Additionally, 
a Flask web application was developed to allow users to input relevant housing details and receive real-time price predictions.

# 🔎 Problem Statement
The objective was to predict housing prices based on multiple factors. Accurate price prediction is crucial for potential buyers, sellers, and real estate agencies to make informed decisions. 
The project handles the complexities of real-world data, including missing values, categorical variables, and data scaling.

# 🛠️ Tools & Technologies Used
Programming Language: Python

Data Analysis & Processing: Pandas, NumPy

Data Visualization: Matplotlib, Seaborn

Machine Learning Model: Scikit-Learn (Random Forest Regressor)

Model Evaluation: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R² Score

Web Development: Flask (for web-based application)

Model Deployment: Joblib (to save and load models)

# 🧹 Data Preprocessing
Handling Missing Values: Missing data was handled using median imputation.

One-Hot Encoding: The categorical feature ocean_proximity was encoded using pd.get_dummies().

Feature Scaling: StandardScaler was applied to normalize numerical features for optimal model performance.

# 🧑‍💻 Model Building
The Random Forest Regressor was chosen due to its ability to handle non-linear relationships and reduce overfitting. Various hyperparameters like n_estimators, max_depth, and min_samples_split were optimized using GridSearchCV.

Model Evaluation Metrics:
Mean Absolute Error (MAE) to measure the average magnitude of errors.

Root Mean Squared Error (RMSE) to assess prediction errors.

R² Score to indicate how well the model explains the variance in the data.

# 🌐 Flask Web Application
A simple and interactive Flask-based web application was developed for real-time house price prediction.

Users input house details via a web form.

The input data is processed and passed to the trained model.

The model predicts the house price, and the result is displayed to the user.

The application ensures proper validation to prevent incorrect inputs and maintains an easy-to-use interface.

# 🏁 Conclusion
The Random Forest model demonstrated a high level of accuracy compared to other models like Linear Regression and KNN.

Train Accuracy: Up to 88%

Test Accuracy: Up to 81%

The model’s robustness makes it ideal for real-world price prediction.

The Flask web application offers users an efficient way to predict housing prices using their own data.

This project showcases end-to-end experience in data preprocessing, feature engineering, model building, evaluation, and deployment using Flask. It highlights practical problem-solving in the domain of real estate price prediction.
