Write-Host "--- Fast Setup & Run ---"
Write-Host "1. Creating local environment to bypass Windows path limits..."
python -m venv venv

Write-Host "2. Installing dependencies in local environment..."
# Upgrade pip first
.\venv\Scripts\python -m pip install --upgrade pip
# Install packages (using local venv drastically creates shorter paths)
.\venv\Scripts\pip install pandas "numpy<2.0.0" scikit-learn streamlit plotly altair
.\venv\Scripts\pip install tensorflow --default-timeout=1000

Write-Host "3. Training Model on Hadoop Logs..."
if (Test-Path "data/hadoop_logs.csv") {
    .\venv\Scripts\python src/train.py --data "data/hadoop_logs.csv"
} else {
    Write-Host "Warning: data/hadoop_logs.csv not found. Using sample data."
    .\venv\Scripts\python src/train.py
}

Write-Host "4. Launching Dashboard..."
Write-Host "If the browser doesn't open, copy the URL below."
.\venv\Scripts\streamlit run app.py
