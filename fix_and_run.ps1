Write-Host "--- Fixing Environment ---"

# 1. Attempt to Enable Long Paths (Needs Admin usually, trying anyway)
Write-Host "Step 1: Enabling Windows Long Path Support..."
$regPath = "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem"
$name = "LongPathsEnabled"
try {
    Set-ItemProperty -Path $regPath -Name $name -Value 1 -ErrorAction Stop
    Write-Host "Success: Long Paths Enabled." -ForegroundColor Green
} catch {
    Write-Host "Warning: Could not automatically enable Long Paths. You may need to run PowerShell as Administrator." -ForegroundColor Yellow
    Write-Host "If the installation fails again with 'OSError', please run this script as Administrator."
}

# 2. Cleanup
Write-Host "Step 2: Cleaning up broken installation..."
pip uninstall -y tensorflow tensorflow-intel protobuf numpy

# 3. Install Correct Versions
Write-Host "Step 3: Installing TensorFlow with correct NumPy..."
pip install "numpy<2.0.0" tensorflow --default-timeout=1000

# 4. Train
Write-Host "Step 4: Running Training..."
python src/train.py --data "data/hadoop_logs.csv"

# 5. Run App
Write-Host "Step 5: Starting App..."
streamlit run app.py
