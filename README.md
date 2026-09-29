# Ames Housing Price Prediction - ML Fundamentals Checkpoint

This project is a machine learning assessment notebook designed to build, train, and evaluate a **Ridge Regression model** to predict home prices (`SalePrice`) using the Ames Housing dataset.

---

##  Tech Stack & Dependencies
The codebase is implemented in **Python 3** within a Jupyter Notebook environment and relies on the following standard libraries:
* **Pandas & NumPy**: For data loading, manipulation, and array formatting.
* **Scikit-Learn (sklearn)**: For the complete predictive modeling pipeline:
  * `model_selection.train_test_split` [1]
  * `preprocessing.StandardScaler` [1]
  * `linear_model.Ridge` [1]
  * `metrics` (`mean_squared_error`, `r2_score`) [1]

---

##  Code Structure & Pipeline

The script processes the data through a 5-step workflow, validated at each stage by integrated `assert` test blocks:

### 1. Data Cleaning & Train-Test Split
* **Filtering**: Loads `ames.csv`, keeping only numeric features and columns with zero missing values.
* **Segmentation**: Splits the dataframe into features (`X`) and the target variable (`y = SalePrice`).
* **Splitting**: Segregates data into training (60%) and testing (40%) sets with a fixed `random_state=42`.

### 2. Data Preprocessing (Scaling)
* Instantiates a `StandardScaler`.
* Fits and transforms the training features (`X_train`).
* Transforms the testing features (`X_test`) independently to prevent data leakage.

### 3. Ridge Model Training
* Fits a `Ridge` regression model using L2 regularization to penalize high coefficients and prevent overfitting.
* **Hyperparameters**: Configured with `alpha=100`, `solver="sag"` (Stochastic Average Gradient descent), and `random_state=1`.

### 4. Performance Evaluation
* Generates predictions for both the scaled training and testing data subsets.
* Computes performance metrics using **Root Mean Squared Error (RMSE)** and **R-squared (R²)**.

### 5. Model Interpretation & Benchmark
* Compares the evaluation metrics against a baseline Ordinary Least Squares (OLS) Linear Regression model:

| Model | Train RMSE | Test RMSE |
| :--- | :---: | :---: |
| **Linear Regression** | \$33,633.14 | \$39,255.80 |
| **Ridge Regression** | \$33,910.84 | \$39,213.66 |

---
