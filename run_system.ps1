Write-Host "--- Log Anomaly Detection System Setup ---"

# 1. Install Dependencies
Write-Host "1/3 Installing Dependencies (TensorFlow)... This may take a while."
pip uninstall -y tensorflow tensorflow-intel protobuf
pip install tensorflow pandas numpy scikit-learn streamlit plotly altair

# 2. Train Model
Write-Host "2/3 Training Model on 'data/hadoop_logs.csv'..."
if (Test-Path "data/hadoop_logs.csv") {
    python src/train.py --data "data/hadoop_logs.csv"
} else {
    Write-Host "Error: data/hadoop_logs.csv not found!"
    exit 1
}

# 3. Run Dashboard
Write-Host "3/3 Starting Dashboard..."
streamlit run app.py
