import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from inference import AnomalyDetectorSystem

st.set_page_config(page_title="Log Anomaly Detection", layout="wide")

st.title("🛡️ Cybersecurity Log Anomaly Detection System")
st.markdown("### Hybrid Unsupervised Approach (Isolation Forest + LSTM Autoencoder)")

# Sidebar
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Log CSV", type=["csv"])

# Load System
@st.cache_resource
def load_system():
    system = AnomalyDetectorSystem(model_dir='models')
    success = system.load_models()
    return system, success

system, loaded = load_system()

if not loaded:
    st.error("Failed to load models. Please ensure 'models/' directory exists and contains trained models.")
    st.info("You may need to run 'python src/train.py' first.")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} logs.")
        
        if st.sidebar.button("Analyze Logs"):
            if not loaded:
                st.error("Models not loaded.")
            else:
                with st.spinner("Analyzing logs (this may take a moment)..."):
                    results = system.predict(df)
                
                # Metrics
                total_logs = len(results)
                anomalies = results[results['Is_Anomaly'] == True]
                num_anomalies = len(anomalies)
                anomaly_rate = (num_anomalies / total_logs) * 100
                
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Logs", total_logs)
                col2.metric("Detected Anomalies", num_anomalies, delta_color="inverse")
                col3.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
                
                st.markdown("---")
                
                # Visualizations
                st.subheader("Anomaly Score Distribution")
                
                tab1, tab2 = st.tabs(["Combined Score", "Component Scores"])
                
                with tab1:
                    fig = px.histogram(results, x="Anomaly_Score", nbins=50, title="Hybrid Anomaly Score Distribution", color="Is_Anomaly")
                    st.plotly_chart(fig, use_container_width=True)
                    
                with tab2:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        fig_if = px.scatter(results, x=results.index, y="IF_Score", color="Is_Anomaly", title="Isolation Forest Scores (Message Content)")
                        st.plotly_chart(fig_if, use_container_width=True)
                    with col_b:
                        fig_lstm = px.scatter(results, x=results.index, y="LSTM_Score", color="Is_Anomaly", title="LSTM Scores (Event Sequence)")
                        st.plotly_chart(fig_lstm, use_container_width=True)
                
                # Detailed Table
                st.subheader("Detected Anomalies")
                if num_anomalies > 0:
                    st.dataframe(anomalies[['Date', 'Time', 'Content', 'EventTemplate', 'Is_Anomaly', 'Anomaly_Score', 'Anomaly_Reason']])
                else:
                    st.info("No anomalies detected.")
                
                # Export
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results CSV", csv, "anomaly_results.csv", "text/csv")
                
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to begin analysis.")
    st.markdown("""
    **Expected CSV Format:**
    - `Date`, `Time`
    - `Content` (Log message)
    - `EventTemplate` (Template of the log)
    - `EventId` (Identifier for the event type)
    """)
