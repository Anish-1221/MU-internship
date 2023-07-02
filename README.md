# Spaceship Andromeda

# Description

The objective of this project is to develop a robust predictive model that accurately determines whether a person, identified by their PassengerId, has been transported to another dimension or not. The dataset contains records of individuals and their associated features, which will be used for data analysis and building predictive models.

# Directory Structure

- Final_Submission.ipynb: Contains the final code of the project.
- train.csv: Contains the training data used in the project.
- test.csv: Contains the testing data used to make final predictions.
- /Submission Files: Contains 8 different files pertaining to different models with/without PCA.

# Dependencies

- pandas
- numpy
- lazypredict
- sklearn
- seaborn
- matplotlib
- lightgbm
- xgboost
- mpl_toolkits

Please note that some of the libraries listed, such as sklearn and matplotlib, are part of the broader scikit-learn and Matplotlib packages, respectively, which you may already have installed as dependencies of other libraries. Additionally, make sure you have the latest versions of these libraries to avoid any compatibility issues.

# Approach

1. **Exploratory Data Analysis (EDA)**

   - Load the dataset using `pd.read_csv()` or any other appropriate method.
   - Explore the structure of the dataset using `df.head()`, `df.nunique()`, `df.isnull().sum()`, and `df.describe()`.
   - Analyze the distribution of the target variable and check for any class imbalance.
   - Visualize the relationships between different features using plots such as histograms and correlation matrices.
   - Identify any missing values or outliers in the dataset.

2. **Preprocessing**

   - Handle missing values: Fill in missing values for numeric columns using the KNN imputation method.
   - Encode categorical variables: Convert categorical variables into numerical representations using techniques like one-hot encoding.

3. **Splitting the Dataset**

   - Split the preprocessed dataset into training and testing sets.

4. **Implementing Machine Learning Models**

   - Use Lazy Classifier in order to implement multiple machine learning models.

5. **Model Evaluation**

   - Instantiate each model with appropriate hyperparameters.
   - Fit the models on the training data using.
   - Make predictions on the test data using.
   - Evaluate the performance of each model using appropriate evaluation metrics such as accuracy.

6. **Grid Search Cross-Validation**

   - Create a parameter grid for each model, specifying the hyperparameters to tune.
   - Get the best hyperparameters.
   - Refit the model with the best hyperparameters on the entire training data.

7. **Feature Dimensionality Reduction (Optional)**

   - Apply dimensionality reduction techniques like Principal Component Analysis (PCA) to reduce the number of features in the dataset. This step can help improve model performance and reduce computational complexity.

8. **Selecting the Best Model**

   - Compare the performance of the models both with and without dimensionality reduction.
   - Select the top performing models based on evaluation metrics, taking into consideration any trade-offs such as run-time.

9. **Predictions**

   - Use the trained models to make predictions on the unseen test data which is later submitted on Kaggle.

# Submission Accuracy:

 **Best accuracy on kaggle was 80.4% from using XGBClassifier without using PCA**
