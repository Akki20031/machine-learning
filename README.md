# Diabetes Prediction Using Machine Learning

This project uses machine learning to predict whether a given patient is diabetic or not based on specific health parameters. The model is built using Python and various libraries such as NumPy, Pandas, scikit-learn for preprocessing, training, and evaluation.

## Problem Statement
Goal: The aim of this project is to develop a machine learning model that can predict whether a patient has diabetes or not based on medical data. This prediction can help in early detection and diagnosis, allowing healthcare providers to take preventive actions and improve patient outcomes.

The dataset used contains medical parameters such as:

Pregnancies
Glucose levels
Blood pressure
Skin thickness
Insulin levels
BMI (Body Mass Index)
Diabetes pedigree function
Age
The objective is to use this data to train a classifier that can predict diabetes.

Dataset
The dataset used for this project contains patient data related to diabetes diagnosis. It consists of various features that are critical for predicting diabetes. The target variable is binary: 1 indicates the patient is diabetic, while 0 indicates non-diabetic.

The dataset can be found here.

## Diagram 
![WhatsApp Image 2024-09-08 at 22 55 41_4b614e7c](https://github.com/user-attachments/assets/6257b21d-e5f9-487a-ad45-a4a7b9018994)


### Steps followed 
Step 1: Import Libraries
The following libraries are imported to handle data loading, preprocessing, and model building:

NumPy: Used for numerical operations.
Pandas: Used for data manipulation and loading.
scikit-learn: Used for data preprocessing, model creation, and evaluation.
![Screenshot 2024-09-05 164811](https://github.com/user-attachments/assets/2ea8de7b-9cf1-4206-9f68-9e4dd23bd721)
Step 2: Load and Preprocess Data
The dataset is loaded using pandas from a CSV file.
Data preprocessing is performed by scaling the features using StandardScaler to ensure that all features contribute equally to the prediction model.
The dataset is split into features (X) and target (y), where X contains the independent variables and y contains the target variable (diabetic or not).
![Screenshot 2024-09-05 165046](https://github.com/user-attachments/assets/333235d8-1927-4af6-861c-54177a175b24)

Step 3: Train-Test Split
The dataset is divided into training and testing sets using train_test_split from scikit-learn. 80% of the data is used for training, and 20% for testing.
![train](https://github.com/user-attachments/assets/a090cd49-b508-4abc-b328-a0dad81f7331)

Step 4: Data Scaling
The feature set is standardized using StandardScaler to scale all variables to a common range, which helps in improving the performance of the model.

![scalar](https://github.com/user-attachments/assets/7b385ed7-0bb7-4d59-9a05-956a4a3b0268)

Step 5: Build the SVM Model
A Support Vector Machine (SVM) classifier is used to train the model. SVM is effective for binary classification problems such as predicting whether a patient is diabetic or not.

Step 6: Model Evaluation
The model’s performance is evaluated on the test data using accuracy score.
![model evaluation](https://github.com/user-attachments/assets/9a0ed2be-6b90-498d-83af-f23e41c8c8c2)

Step 7: Predicting New Data
Once the model is trained and tested, it can be used to predict the outcome for new patient data.
![predictive](https://github.com/user-attachments/assets/4bbc256e-44c9-4d3b-ab3d-11d264712e87)


Requirements
The project uses the following dependencies:

Python 3.x

NumPy

Pandas

scikit-learn

Usage:

Once you have the project set up, you can train and test the model by running the diabetes_prediction.py script.

Insights and Model Performance:

Accuracy: The accuracy score on the test data shows how well the model performs in predicting diabetes. Further improvements could be made by tuning hyperparameters or trying different models.


Future Enhancements:

Hyperparameter Tuning: Apply GridSearchCV or RandomizedSearchCV to find the optimal parameters for the SVM model.

Model Comparison: Compare the performance of different machine learning models like Logistic Regression, Random Forest, or K-Nearest Neighbors (KNN).

Cross-Validation: Use k-fold cross-validation to evaluate the robustness of the model.

Feature Engineering: Apply feature engineering techniques to extract more meaningful insights from the data.
