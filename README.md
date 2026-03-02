# 🛡️ Cybersecurity Log Anomaly Detection System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A hybrid unsupervised machine learning system for detecting anomalies in cybersecurity logs using Isolation Forest and LSTM Autoencoder with an interactive Streamlit dashboard.

## 🎯 Overview

This project combines two complementary machine learning approaches to detect anomalies in log files:

- **Message-Level Detection**: Uses Isolation Forest on TF-IDF vectors to identify unusual log content
- **Sequence-Level Detection**: Employs LSTM Autoencoder to detect abnormal event patterns over time
- **Hybrid Scoring**: Intelligently combines both models for accurate anomaly detection

## ✨ Features

- ✅ Hybrid ML approach (Isolation Forest + LSTM)
- ✅ Real-time anomaly detection
- ✅ Interactive web dashboard with visualizations
- ✅ Pre-trained on Hadoop logs
- ✅ Explainable AI - provides reasons for anomalies
- ✅ Easy deployment with automation scripts
- ✅ Supports custom log formats
- ✅ Export results to CSV

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Log Input (CSV)                      │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼─────┐          ┌─────▼────┐
    │ TF-IDF   │          │ Sequence │
    │ Features │          │ Creation │
    └────┬─────┘          └─────┬────┘
         │                      │
    ┌────▼─────────┐      ┌────▼──────────┐
    │  Isolation   │      │     LSTM      │
    │   Forest     │      │  Autoencoder  │
    └────┬─────────┘      └────┬──────────┘
         │                     │
         └──────────┬──────────┘
                    │
            ┌───────▼────────┐
            │ Hybrid Scoring │
            └───────┬────────┘
                    │
            ┌───────▼────────┐
            │   Dashboard    │
            └────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation & Running

**Option 1: Quick Setup (Recommended for Windows)**
```powershell
.\quick_setup.ps1
```

**Option 2: Manual Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python src/train.py --data data/hadoop_logs.csv

# Run dashboard
streamlit run app.py
```

**Option 3: Fix Environment Issues**
```powershell
.\fix_and_run.ps1
```

### Access the Dashboard

Once running, open your browser and navigate to:
- Local: http://localhost:8501

## 📊 Usage

1. **Upload Log File**: Click "Upload Log CSV" in the sidebar
2. **Analyze**: Click "Analyze Logs" button
3. **View Results**: 
   - See anomaly metrics and statistics
   - Explore visualizations of anomaly scores
   - Review detected anomalies with explanations
4. **Export**: Download results as CSV

### Expected CSV Format

Your log file should contain these columns:
- `Date` - Log date
- `Time` - Log timestamp
- `Content` - Log message content
- `EventTemplate` - Template of the log event
- `EventId` - Event identifier

## 📁 Project Structure

```
cybersecurity-log-anomaly-detection/
├── src/
│   ├── preprocess.py      # Data loading and cleaning
│   ├── features.py        # TF-IDF feature extraction
│   ├── models.py          # ML models (Isolation Forest + LSTM)
│   ├── train.py           # Training pipeline
│   └── inference.py       # Prediction system
├── data/
│   ├── hadoop_logs.csv    # Training data
│   └── sample_logs.csv    # Sample data
├── models/
│   ├── isolation_forest.pkl
│   ├── lstm_model.h5
│   ├── tfidf_vectorizer.pkl
│   └── event_mapping.pkl
├── app.py                 # Streamlit dashboard
├── requirements.txt       # Dependencies
├── quick_setup.ps1        # Automated setup script
├── run_system.ps1         # Standard run script
└── fix_and_run.ps1        # Environment fix script
```

## 🔧 Configuration

### Model Parameters

Edit `src/train.py` to adjust:
- `contamination`: Expected anomaly percentage (default: 0.1)
- `sequence_length`: LSTM window size (default: 3)
- `max_features`: TF-IDF vocabulary size (default: 1000)

### Anomaly Threshold

Edit `src/inference.py`:
- `threshold_fixed`: Anomaly score cutoff (default: 0.6)

## 🧪 Training on Custom Data

```bash
python src/train.py --data path/to/your/logs.csv
```

Ensure your CSV has the required columns: `Date`, `Time`, `Content`, `EventTemplate`, `EventId`

## 📈 Evaluation Metrics

The system provides:
- **Anomaly Score**: Hybrid score (0-1 range)
- **Anomaly Rate**: Percentage of flagged logs
- **Component Scores**: Individual IF and LSTM scores
- **Anomaly Reasons**: Explanations for detections

## 🛠️ Tech Stack

- **Backend**: Python 3.8+
- **ML Frameworks**: TensorFlow, scikit-learn
- **NLP**: TF-IDF Vectorization
- **Frontend**: Streamlit
- **Visualization**: Plotly, Altair
- **Data Processing**: Pandas, NumPy

## 👥 Team Contributions

This project is designed for collaborative development:

- **Member 1**: Data Preprocessing & Feature Engineering
- **Member 2**: Isolation Forest Implementation
- **Member 3**: LSTM Autoencoder Implementation
- **Member 4**: Integration, Training & Dashboard

See `TEAM_DIVISION.md` for detailed file assignments.

## 🎓 Use Cases

- Security Operations Center (SOC) log monitoring
- Intrusion detection systems
- System health monitoring
- DevOps anomaly detection
- Compliance and audit log analysis
- Research in log analysis and ML

## 🔒 Security Considerations

- Models are trained on historical data
- No sensitive data is stored permanently
- All processing is done locally
- Consider adding authentication for production use

## 🐛 Troubleshooting

**Issue: TensorFlow/NumPy compatibility error**
```powershell
.\fix_and_run.ps1
```

**Issue: Windows path length errors**
```powershell
.\quick_setup.ps1  # Uses local venv with shorter paths
```

**Issue: Models not found**
```bash
python src/train.py --data data/hadoop_logs.csv
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

Project Link: [https://github.com/phani-18-coder/cybersecurity-log-anomaly-detection](https://github.com/phani-18-coder/cybersecurity-log-anomaly-detection)

## 🙏 Acknowledgments

- Hadoop log dataset for training
- Streamlit for the amazing dashboard framework
- TensorFlow and scikit-learn communities

---

⭐ If you find this project useful, please consider giving it a star!
