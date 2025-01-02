import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Style and Page Configuration
def set_page_style():
    st.set_page_config(
        page_title="Fraud Detection Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stAlert {
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .best-model-badge {
            background-color: #28a745;
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.8rem;
            margin-left: 0.5rem;
        }
        .feature-importance-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            border: 1px solid #e9ecef;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding-right: 1rem;
            padding-left: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

# Model Loading Function
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Visualization Functions
def create_metrics_chart(metrics):
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'ROC-AUC', 'Precision', 'Recall', 'F1-Score'],
        'Value': [
            metrics['Accuracy'],
            metrics['ROC-AUC'],
            metrics['Precision'],
            metrics['Recall'],
            metrics['F1-Score']
        ]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics_df['Metric'],
        y=metrics_df['Value'],
        marker_color=['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728'],
    ))
    
    fig.update_layout(
        title='Model Performance Metrics',
        yaxis_title='Score',
        yaxis_range=[0, 1],
        showlegend=False,
        height=300,
    )
    
    return fig

def create_confusion_matrix(metrics):
    z = [[metrics['True Negatives'], metrics['False Positives']],
         [metrics['False Negatives'], metrics['True Positives']]]
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        text=z,
        texttemplate="%{text}",
        textfont={"size": 12},
        colorscale='RdYlBu'
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        height=300,
    )
    
    return fig

def create_feature_importance_plot(model):
    feature_importance = pd.DataFrame({
        'feature': [f'V{i}' for i in range(1, 29)] + ['Amount'],
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=feature_importance['importance'],
        y=feature_importance['feature'],
        orientation='h',
        marker_color='#28a745'
    ))
    
    fig.update_layout(
        title='Top 10 Most Important Features',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_threshold_analysis_plot(probabilities, true_labels=None):
    thresholds = np.linspace(0, 1, 100)
    predictions = np.array([probabilities >= threshold for threshold in thresholds])
    fraud_rates = predictions.mean(axis=1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=fraud_rates,
        mode='lines',
        name='Fraud Rate',
        line=dict(color='#28a745')
    ))
    
    fig.add_vline(x=0.997589, line_dash="dash", line_color="red",
                  annotation_text="Optimal Threshold")
    
    fig.update_layout(
        title='Threshold Analysis',
        xaxis_title='Probability Threshold',
        yaxis_title='Fraud Rate',
        height=300
    )
    
    return fig

def display_fraud_summary(results_df):
    total_transactions = len(results_df)
    fraud_cases = len(results_df[results_df['Prediction'] == 1])
    fraud_percentage = (fraud_cases / total_transactions) * 100 if total_transactions > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", f"{total_transactions:,}")
    with col2:
        st.metric("Predicted Fraudulent", f"{fraud_cases:,}")
    with col3:
        st.metric("Fraud Rate", f"{fraud_percentage:.2f}%")

def process_transactions(input_data, model, selected_model_info):
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1]
    
    results_df = input_data.copy()
    results_df['Prediction'] = predictions
    results_df['Fraud Probability'] = probabilities
    results_df['Above Threshold'] = probabilities >= selected_model_info['metrics']['Optimal Threshold']
    
    return results_df

# Main Application
def main():
    set_page_style()
    
    # Model Configurations
    models = {
        'Logistic Regression': {
            'path': 'models/logistic_regression_model.pkl',
            'description': 'High accuracy with good recall',
            'metrics': {
                'Accuracy': 0.998859,
                'ROC-AUC': 0.983626,
                'Precision': 0.615385,
                'Recall': 0.897959,
                'F1-Score': 0.730290,
                'Optimal Threshold': 1.000000,
                'True Negatives': 56809.0,
                'False Positives': 55.0,
                'False Negatives': 10.0,
                'True Positives': 88.0
            }
        },
        'Decision Tree': {
            'path': 'models/decision_tree_model.pkl',
            'description': 'Simple and interpretable model',
            'metrics': {
                'Accuracy': 0.997577,
                'ROC-AUC': 0.942761,
                'Precision': 0.406542,
                'Recall': 0.887755,
                'F1-Score': 0.557692,
                'Optimal Threshold': 1.000000,
                'True Negatives': 56737.0,
                'False Positives': 127.0,
                'False Negatives': 11.0,
                'True Positives': 87.0
            }
        },
        'Random Forest': {
            'path': 'models/random_forest_model.pkl',
            'description': 'Highest precision with balanced performance',
            'metrics': {
                'Accuracy': 0.999175,
                'ROC-AUC': 0.977802,
                'Precision': 0.981132,
                'Recall': 0.530612,
                'F1-Score': 0.688742,
                'Optimal Threshold': 0.750000,
                'True Negatives': 56863.0,
                'False Positives': 1.0,
                'False Negatives': 46.0,
                'True Positives': 52.0
            }
        },
        'XGBoost': {
            'path': 'models/xgboost_model.pkl',
            'description': 'Best overall performance',
            'metrics': {
                'Accuracy': 0.999596,
                'ROC-AUC': 0.984043,
                'Precision': 0.912088,
                'Recall': 0.846939,
                'F1-Score': 0.878307,
                'Optimal Threshold': 0.997589,
                'True Negatives': 56856.0,
                'False Positives': 8.0,
                'False Negatives': 15.0,
                'True Positives': 83.0
            }
        },
        'Support Vector Machine': {
            'path': 'models/svm_model.pkl',
            'description': 'Strong performance on high-dimensional data',
            'metrics': {
                'Accuracy': 0.999280,
                'ROC-AUC': 0.984772,
                'Precision': 0.752212,
                'Recall': 0.867347,
                'F1-Score': 0.805687,
                'Optimal Threshold': 0.994470,
                'True Negatives': 56836.0,
                'False Positives': 28.0,
                'False Negatives': 13.0,
                'True Positives': 85.0
            }
        },
        'LightGBM': {
            'path': 'models/lightgbm_model.pkl',
            'description': 'Fast and efficient gradient boosting',
            'metrics': {
                'Accuracy': 0.999526,
                'ROC-AUC': 0.964819,
                'Precision': 0.881720,
                'Recall': 0.836735,
                'F1-Score': 0.858639,
                'Optimal Threshold': 0.993473,
                'True Negatives': 56853.0,
                'False Positives': 11.0,
                'False Negatives': 16.0,
                'True Positives': 82.0
            }
        },
        'CatBoost': {
            'path': 'models/catboost_model.pkl',
            'description': 'Excellent handling of categorical features',
            'metrics': {
                'Accuracy': 0.999526,
                'ROC-AUC': 0.981731,
                'Precision': 0.898876,
                'Recall': 0.816327,
                'F1-Score': 0.855615,
                'Optimal Threshold': 0.957154,
                'True Negatives': 56855.0,
                'False Positives': 9.0,
                'False Negatives': 18.0,
                'True Positives': 80.0
            }
        }
    }
    
    # Application Header
    st.title('🔍 Credit Card Fraud Detection Dashboard')
    st.markdown("""
        This advanced fraud detection system uses state-of-the-art machine learning models to identify 
        potentially fraudulent credit card transactions. Select a model and upload your transaction data 
        for instant analysis.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Selection")
        
        model_options = list(models.keys())
        default_index = model_options.index('XGBoost')
        
        selected_model = st.selectbox(
            'Select Model',
            model_options,
            index=default_index,
            format_func=lambda x: f"{x} {'🏆 Best Model' if x == 'XGBoost' else ''}",
            help="Choose the machine learning model for fraud detection"
        )
        
        if selected_model == 'XGBoost':
            st.markdown("""
                <div style='background-color: #d4edda; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
                    <h4 style='color: #155724; margin: 0;'>🏆 Best Performing Model</h4>
                    <p style='color: #155724; margin-top: 0.5rem;'>
                        XGBoost achieves the highest overall performance with:
                        • 99.96% Accuracy
                        • 91.21% Precision
                        • 84.69% Recall
                        • 87.83% F1-Score
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        metrics_tab1, metrics_tab2 = st.tabs(["📈 Performance", "🔢 Confusion Matrix"])
        
        with metrics_tab1:
            st.plotly_chart(
                create_metrics_chart(models[selected_model]['metrics']),
                use_container_width=True
            )
        
        with metrics_tab2:
            st.plotly_chart(
                create_confusion_matrix(models[selected_model]['metrics']),
                use_container_width=True
            )
        
        st.markdown(f"**Optimal Threshold:** {models[selected_model]['metrics']['Optimal Threshold']:.4f}")
    
    # XGBoost Exclusive Features
    if selected_model == 'XGBoost':
        st.markdown("### 🔍 XGBoost Exclusive Features")
        
        tab1, tab2, tab3 = st.tabs(["📊 Feature Importance", "📈 Threshold Analysis", "🎯 Model Insights"])
        
        with tab1:
            model = load_model(models['XGBoost']['path'])
            if model is not None:
                st.plotly_chart(create_feature_importance_plot(model), use_container_width=True)
                
                st.markdown("""
                    <div class='feature-importance-card'>
                        <h4>Understanding Feature Importance</h4>
                        <p>The feature importance plot shows which transaction characteristics are most influential 
                        in detecting fraud. Higher scores indicate stronger predictive power.</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.plotly_chart(create_threshold_analysis_plot([0.1, 0.2, 0.3]), use_container_width=True)
            
            st.markdown("""
                <div class='feature-importance-card'>
                    <h4>Threshold Optimization</h4>
                    <p>The optimal threshold of 0.997589 was determined through extensive validation to maximize 
                    model performance. This threshold provides the best balance between fraud detection and false alarms.</p>
                </div>
            """, unsafe_allow_html=True)
            
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                    <div class='feature-importance-card'>
                        <h4>💪 Strengths</h4>
                        <ul>
                            <li>Highest accuracy among all models (99.96%)</li>
                            <li>Excellent precision-recall balance</li>
                            <li>Robust to outliers</li>
                            <li>Low false positive rate</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class='feature-importance-card'>
                        <h4>📈 Best Use Cases</h4>
                        <ul>
                            <li>High-volume transaction processing</li>
                            <li>Real-time fraud detection</li>
                            <li>Complex pattern recognition</li>
                            <li>Handling imbalanced data</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
    
    # Data Input Section
    st.markdown("---")
    st.header("📥 Transaction Data Input")
    
    input_tab1, input_tab2 = st.tabs(["📁 Upload CSV", "✏️ Paste Data"])
    
    input_data = None
    
    with input_tab1:
        st.markdown("""
            Upload a CSV file containing transaction data. The file should include the following features:
            - V1 through V28 (normalized transaction features)
            - Amount (transaction amount)
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with transaction features"
        )
        
        if uploaded_file:
            try:
                input_data = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(input_data):,} transactions")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with input_tab2:
        st.markdown("""
            Paste your transaction data in CSV format. Each row should contain:
            - V1 through V28 (normalized transaction features)
            - Amount (transaction amount)
        """)
        
        csv_input = st.text_area(
            "Input Transactions",
            height=150,
            placeholder="V1,V2,V3,...,V28,Amount",
            help="Paste CSV formatted transaction data"
        )
        
        if csv_input.strip():
            try:
                input_data = pd.read_csv(io.StringIO(csv_input), header=None)
                input_data.columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
                st.success(f"Successfully parsed {len(input_data):,} transactions")
            except Exception as e:
                st.error(f"Error parsing input: {str(e)}")
    
    # Analysis Section
    if st.button("🔍 Analyze Transactions", type="primary", use_container_width=True):
        if input_data is not None:
            with st.spinner("Analyzing transactions..."):
                model = load_model(models[selected_model]['path'])
                
                if model is not None:
                    # Process transactions
                    results_df = process_transactions(input_data, model, models[selected_model])
                    
                    st.markdown("---")
                    st.header("📊 Analysis Results")
                    
                    # Display summary metrics
                    display_fraud_summary(results_df)
                    
                    # Results Tabs
                    results_tab1, results_tab2, results_tab3 = st.tabs([
                        "📋 Detailed Results",
                        "📊 Visualization",
                        "⚠️ High Risk Transactions"
                    ])
                    
                    with results_tab1:
                        st.dataframe(
                            results_df.style.background_gradient(
                                subset=['Fraud Probability'],
                                cmap='RdYlBu_r'
                            ),
                            height=400
                        )
                    
                    with results_tab2:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Probability distribution
                            fig = px.histogram(
                                results_df,
                                x='Fraud Probability',
                                title='Distribution of Fraud Probabilities',
                                nbins=50
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Amount vs Probability scatter
                            fig = px.scatter(
                                results_df,
                                x='Amount',
                                y='Fraud Probability',
                                color='Prediction',
                                title='Transaction Amount vs Fraud Probability'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with results_tab3:
                        # Filter high-risk transactions
                        high_risk = results_df[
                            results_df['Fraud Probability'] >= models[selected_model]['metrics']['Optimal Threshold']
                        ].sort_values('Fraud Probability', ascending=False)
                        
                        if len(high_risk) > 0:
                            st.warning(f"Found {len(high_risk):,} high-risk transactions")
                            st.dataframe(
                                high_risk.style.background_gradient(
                                    subset=['Fraud Probability'],
                                    cmap='RdYlBu_r'
                                ),
                                height=400
                            )
                        else:
                            st.success("No high-risk transactions detected")
                    
                    # Export Results
                    st.markdown("---")
                    st.subheader("📥 Export Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Excel export
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            results_df.to_excel(writer, sheet_name='All Transactions', index=False)
                            if len(high_risk) > 0:
                                high_risk.to_excel(writer, sheet_name='High Risk', index=False)
                        
                        st.download_button(
                            label="📥 Download Excel Report",
                            data=output.getvalue(),
                            file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            help="Download complete analysis results in Excel format"
                        )
                    
                    with col2:
                        # CSV export
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV Results",
                            data=csv,
                            file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Download complete analysis results in CSV format"
                        )
        else:
            st.error("Please provide transaction data to analyze")

if __name__ == '__main__':
    main()