# Infosys Springboard 5.0 - Fraud Detection System

This project provides a robust, interactive dashboard for detecting credit card fraud using various machine learning models. The application is designed to streamline the analysis of transaction data and identify potential fraudulent activities.

## Features

- **Fraud Detection Models**: Includes Logistic Regression, Decision Tree, Random Forest, XGBoost, Support Vector Machine, LightGBM, and CatBoost.
- **Performance Metrics Visualization**: Detailed analysis of model accuracy, ROC-AUC, precision, recall, and F1-score.
- **Threshold Analysis**: Helps determine the optimal probability threshold for fraud classification.
- **Feature Importance**: Highlights the most impactful features for fraud prediction.
- **Transaction Input Options**: Upload data via CSV or paste transaction data directly into the app.
- **Export Results**: Download analysis reports in Excel or CSV format.

## Contributors

- [Vishal Tyagi](https://github.com/vishalxtyagi)
- [Suhas S](https://github.com/suhass434)
- [Prasanth Kumar](https://github.com/prasanth1221)
- [Vidhita Tutu](https://github.com/VidhitaTutu30)
- [Dhanush Tadisetti](https://github.com/Dhanushtadisetti)
- [Prateek Mishra](https://github.com/PrateekMishraGithub)
- [Rithik](https://github.com/rithik423)

## How It Works

1. **Model Selection**: Choose a pre-trained model optimized for detecting fraud.
2. **Upload Data**: Provide transaction data either by uploading a CSV file or pasting the data in CSV format.
3. **Analysis**:
   - View metrics such as confusion matrix, ROC-AUC, and feature importance.
   - Analyze fraud probabilities with visualizations for threshold optimization.
   - Identify high-risk transactions.
4. **Export Results**: Save your analysis for further investigation or reporting.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vishalxtyagi/fraud-detection-system
   cd fraud-detection-system
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Models Used

### Logistic Regression
- **Accuracy**: 99.88%
- **F1-Score**: 73.03%
- **Strengths**: High recall for fraud detection.

### XGBoost (🏆 Best Model)
- **Accuracy**: 99.96%
- **F1-Score**: 87.83%
- **Optimal Threshold**: 0.997589
- **Strengths**: Excellent balance of precision and recall with robust performance.

### Other Models
- **Random Forest**: High precision and low false positives.
- **Support Vector Machine**: Effective for high-dimensional data.
- **LightGBM & CatBoost**: Fast and efficient gradient boosting algorithms.

## Input Data Format

The input data should have the following features:

- `V1` to `V28`: Normalized transaction attributes.
- `Amount`: Transaction amount.

Sample Data:
```csv
V1,V2,V3,...,V28,Amount
-1.359807,-0.072781,2.536347,...,-0.021053,149.62
1.191857,0.266151,0.166480,...,0.014724,2.69
```

## Results

- **Detailed Analysis**: Confusion matrix, fraud rate, and feature importance.
- **Visualization**: Fraud probability distribution and high-risk transaction insights.

## References

- Dataset: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Tools: Streamlit, Plotly, Scikit-learn, XGBoost, LightGBM, CatBoost.
