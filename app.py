import streamlit as st
import pandas as pd
import joblib
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def set_page_style():
    st.set_page_config(
        page_title="Fraud Detection Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS to improve the UI
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stAlert {
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .fraud-metric {
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_metrics_chart(metrics):
    fig = go.Figure()
    
    metrics_df = pd.DataFrame({
        'Metric': ['ROC-AUC', 'Precision', 'Recall'],
        'Value': [
            metrics['ROC-AUC'],
            metrics['Precision'],
            metrics['Recall']
        ]
    })
    
    fig.add_trace(go.Bar(
        x=metrics_df['Metric'],
        y=metrics_df['Value'],
        marker_color=['#1f77b4', '#2ca02c', '#ff7f0e'],
    ))
    
    fig.update_layout(
        title='Model Performance Metrics',
        yaxis_title='Score',
        yaxis_range=[0, 1],
        showlegend=False,
        height=300,
    )
    
    return fig

def display_fraud_summary(results_df):
    total_transactions = len(results_df)
    fraud_cases = results_df[results_df['Prediction'] == 1]
    fraud_count = len(fraud_cases)
    fraud_percentage = (fraud_count / total_transactions) * 100 if total_transactions > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Transactions", total_transactions)
    with col2:
        st.metric("Fraud Cases", fraud_count)
    with col3:
        st.metric("Fraud Percentage", f"{fraud_percentage:.2f}%")

def create_probability_histogram(probabilities):
    fig = px.histogram(
        probabilities,
        nbins=50,
        title="Distribution of Fraud Probabilities",
        labels={'value': 'Fraud Probability', 'count': 'Number of Transactions'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(height=300)
    return fig

def main():
    set_page_style()
    
    st.title('🔍 Credit Card Fraud Detection Dashboard')
    st.markdown("""
        This dashboard helps you detect potentially fraudulent credit card transactions 
        using various machine learning models. Upload your transaction data and get instant predictions.
    """)
    
    models = {
        'Random Forest': {
            'path': 'models/random_forest_model.joblib',
            'description': 'Balanced performance with good precision and recall',
            'metrics': {'ROC-AUC': 0.9668, 'Precision': 0.90, 'Recall': 0.85}
        },
        'XGBoost': {
            'path': 'models/xgboost_model.pkl',
            'description': 'High performance with gradient boosting',
            'metrics': {'ROC-AUC': 0.9756, 'Precision': 0.83, 'Recall': 0.86}
        },
        'CatBoost': {
            'path': 'models/catboost_model.pkl',
            'description': 'Handles categorical features well',
            'metrics': {'ROC-AUC': 0.9858, 'Precision': 0.57, 'Recall': 0.87}
        },
        'LightGBM': {
            'path': 'models/lightgbm_model.pkl',
            'description': 'Fast and efficient gradient boosting',
            'metrics': {'ROC-AUC': 0.9600, 'Precision': 0.68, 'Recall': 0.85}
        }
    }
    
    with st.sidebar:
        st.header("📊 Model Configuration")
        selected_model = st.selectbox(
            'Select Model',
            list(models.keys()),
            help="Choose the machine learning model for fraud detection"
        )
        
        st.markdown("---")
        st.markdown(f"**Model Description:**")
        st.markdown(models[selected_model]['description'])
        
        st.plotly_chart(
            create_metrics_chart(models[selected_model]['metrics']),
            use_container_width=True
        )
    
    st.markdown("---")
    
    # File upload section with tabs
    tab1, tab2 = st.tabs(["📁 Upload CSV", "✏️ Paste Data"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload your transaction data (CSV)",
            type=['csv'],
            help="Upload a CSV file with transaction features"
        )
        input_data = None
        if uploaded_file:
            try:
                input_data = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with tab2:
        csv_input = st.text_area(
            "Paste transaction data",
            height=150,
            placeholder="V1,V2,V3,...,V28,Amount",
            help="Paste CSV formatted transaction data"
        )
        if csv_input.strip():
            try:
                input_data = pd.read_csv(io.StringIO(csv_input), header=None)
            except Exception as e:
                st.error(f"Error parsing input: {str(e)}")
    
    if st.button("🔍 Analyze Transactions", type="primary"):
        if input_data is not None:
            with st.spinner("Analyzing transactions..."):
                model = load_model(models[selected_model]['path'])
                
                if model is not None:
                    predictions = model.predict(input_data)
                    probabilities = model.predict_proba(input_data)[:, 1]
                    
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")
                    
                    results_df = input_data.copy()
                    results_df['Prediction'] = predictions
                    results_df['Fraud Probability'] = probabilities
                    
                    display_fraud_summary(results_df)
                    
                    # Visualization tabs
                    viz_tab1, viz_tab2 = st.tabs(["📊 Probability Distribution", "📋 Detailed Results"])
                    
                    with viz_tab1:
                        st.plotly_chart(
                            create_probability_histogram(probabilities),
                            use_container_width=True
                        )
                    
                    with viz_tab2:
                        st.dataframe(
                            results_df.style.background_gradient(
                                subset=['Fraud Probability'],
                                cmap='RdYlGn_r'
                            ),
                            height=400
                        )
                    
                    # Download results
                    if len(results_df) > 0:
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            results_df.to_excel(writer, index=False)
                        
                        st.download_button(
                            label="📥 Download Results",
                            data=output.getvalue(),
                            file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
        else:
            st.error("Please provide transaction data to analyze")

if __name__ == '__main__':
    main()