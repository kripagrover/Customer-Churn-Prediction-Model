# Customer Churn Prediction Model

A machine learning project that predicts whether a telecom customer will churn (leave the service) using the IBM Telco Customer Churn dataset. The pipeline covers exploratory data analysis, preprocessing, handling class imbalance, model comparison, and deployment-ready model serialization.

## Overview

Customer churn is a major concern for subscription-based businesses. This project builds a classification model to identify customers at risk of churning, enabling proactive retention strategies.

The workflow includes:

- **Data exploration** — distributions, correlations, and categorical feature analysis
- **Preprocessing** — label encoding, missing value handling, and train/test split
- **Class imbalance handling** — SMOTE (Synthetic Minority Oversampling Technique)
- **Model comparison** — Decision Tree, Random Forest, and XGBoost with 5-fold cross-validation
- **Model selection** — Random Forest (best default-parameter performance)
- **Inference** — saved model and encoders for single-customer predictions



## Dataset

This project uses the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).


| Detail   | Value                            |
| -------- | -------------------------------- |
| Records  | ~7,043 customers                 |
| Features | 19 (after dropping `customerID`) |
| Target   | `Churn` (Yes / No)               |


**Key features:** tenure, contract type, payment method, monthly/total charges, internet/phone services, and related subscription options.

## Tech Stack

- **Python 3**
- **pandas** & **NumPy** — data manipulation
- **matplotlib** & **seaborn** — visualization
- **scikit-learn** — preprocessing, models, and evaluation
- **imbalanced-learn** — SMOTE oversampling
- **XGBoost** — gradient boosting classifier
- **pickle** — model and encoder persistence



## Project Structure

```
Customer_Churn_PredictionModel/
├── app.py                                        # Streamlit web frontend
├── predict.py                                    # Model loading & inference
├── customer_churn_prediction_model_using_ml.py   # Full ML pipeline
├── requirements.txt
├── README.md
├── .streamlit/config.toml                        # App theme
├── encoders.pkl                                    # Generated after training
└── customer_churn_model.pkl                        # Generated after training
```



## Getting Started



### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Customer_Churn_PredictionModel.git
cd Customer_Churn_PredictionModel
```



### 2. Install dependencies

```bash
pip install -r requirements.txt
```



### 3. Download the dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the project directory.

Update the data path in the script (line 28) from the Colab default:

```python
# Change this:
df = pd.read_csv("/content/WA_Fn-UseC_-Telco-Customer-Churn.xls")

# To your local path, e.g.:
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
```



### 4. Run the pipeline

```bash
python customer_churn_prediction_model_using_ml.py
```

After training, the script saves:

- `customer_churn_model.pkl` — trained Random Forest model and feature names
- `encoders.pkl` — label encoders for categorical columns

### 5. Launch the web app

Once the model files exist, start the Streamlit frontend:

```bash
streamlit run app.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`).

**App features:**

- Tabbed form for demographics, services, and billing details
- One-click sample customer profile
- Churn vs. retention probability chart
- Risk tier (Low / Medium / High) with retention tips



## Pipeline Summary



### Data Preprocessing

1. Drop `customerID` (not useful for modeling)
2. Replace blank `TotalCharges` values with `0.0`
3. Label-encode categorical features and the target (`Churn`: Yes → 1, No → 0)
4. Split data 80/20 (train/test) with `random_state=42`
5. Apply SMOTE on the training set to address class imbalance



### Models Evaluated


| Model         | Method                   |
| ------------- | ------------------------ |
| Decision Tree | `DecisionTreeClassifier` |
| Random Forest | `RandomForestClassifier` |
| XGBoost       | `XGBRFClassifier`        |


Each model is evaluated with **5-fold cross-validation** on SMOTE-balanced training data. **Random Forest** achieves the highest cross-validation accuracy with default hyperparameters and is used for final evaluation on the held-out test set.

### Evaluation Metrics

The final model is assessed using:

- Accuracy score
- Confusion matrix
- Classification report (precision, recall, F1-score)



## Making Predictions

The script includes an example inference block that:

1. Loads `customer_churn_model.pkl` and `encoders.pkl`
2. Encodes a sample customer record
3. Returns a churn prediction and probability

Example output:

```
Prediction: No Churn
Prediction Probability: [prob_no_churn, prob_churn]
```



## Key Insights from EDA

- **Class imbalance** — more customers stay than churn; SMOTE is applied during training
- **Tenure & charges** — longer tenure and higher total charges often correlate with lower churn
- **Contract type** — month-to-month contracts show higher churn rates
- **Services** — customers without online security or tech support tend to churn more



## Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Feature engineering (e.g., tenure bins, charge ratios)
- Threshold optimization for recall on the churn class
- Convert the Colab notebook export into modular Python scripts



## Author

Kripa Grover

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Dataset: IBM Telco Customer Churn (via Kaggle)
- Originally developed in Google Colab

