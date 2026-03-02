import pandas as pd
import numpy as np
import os
from preprocess import preprocess_logs, create_sequences
from features import FeatureExtractor
from models import MessageAnomalyDetector, SequenceAnomalyDetector, to_one_hot
import joblib

# Paths
DATA_PATH = os.path.join('data', 'sample_logs.csv')
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def train_pipeline(data_path):
    print("--- Starting Training Pipeline ---")
    print(f"Training using data from: {data_path}")
    
    # 1. Load and Preprocess Data
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    df = preprocess_logs(df)
    print("Data loaded and preprocessed.")
    
    # 2. NLP Feature Extraction (Message Level)
    print("Extracting NLP features...")
    feature_extractor = FeatureExtractor(max_features=1000)
    # Using 'CleanedText' for training
    tfidf_vectors = feature_extractor.fit_transform(df['CleanedText'])
    feature_extractor.save(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    
    # 3. Message-Level Anomaly Detection (Isolation Forest)
    print("Training Isolation Forest...")
    iso_forest = MessageAnomalyDetector(contamination=0.1) # Assuming 10% anomalies for this synthetic set or just to have some variance
    iso_forest.fit(tfidf_vectors)
    iso_forest.save(os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
    
    # 4. Sequence-Level Anomaly Detection (LSTM)
    print("Preparing sequences for LSTM...")
    # Map EventId to Integers
    event_ids = df['EventId'].unique()
    event_to_int = {e: i for i, e in enumerate(event_ids)}
    int_to_event = {i: e for i, e in enumerate(event_ids)}
    
    # Save mapping
    joblib.dump(event_to_int, os.path.join(MODEL_DIR, 'event_mapping.pkl'))
    
    # Add int column
    df['EventInt'] = df['EventId'].map(event_to_int)
    
    # Create sequences
    SEQUENCE_LENGTH = 3 # Small window for this small data
    vocab_size = len(event_ids)
    
    # Ensure we have enough data
    if len(df) <= SEQUENCE_LENGTH:
        print("Not enough data for sequence generation.")
        return

    sequences = create_sequences(df, window_size=SEQUENCE_LENGTH)
    # Convert sequences (list of lists) to numpy array of integers
    # create_sequences returns raw values. We need to fetch EventInt values manually since create_sequences might have used the original column.
    # Let's re-implement sequence creation here briefly or rely on the function if it handles column selection?
    # preprocess.py's create_sequences took 'EventId' values. If we pass the DF with 'EventInt' renamed to 'EventId' or just extract manually.
    
    # Let's do it manually here to be safe and use EventInt
    raw_sequences = []
    values = df['EventInt'].values
    for i in range(len(values) - SEQUENCE_LENGTH + 1):
        raw_sequences.append(values[i:i+SEQUENCE_LENGTH])
        
    X_seq = np.array(raw_sequences)
    
    # One-Hot Encode for Model
    X_one_hot = to_one_hot(X_seq, vocab_size)
    
    print(f"Training LSTM on {len(X_seq)} sequences of length {SEQUENCE_LENGTH}...")
    lstm_model = SequenceAnomalyDetector(vocab_size=vocab_size, sequence_length=SEQUENCE_LENGTH)
    lstm_model.fit(X_one_hot, epochs=50, batch_size=2) # Small batch for small data
    lstm_model.save(os.path.join(MODEL_DIR, 'lstm_model.h5'))
    
    print("--- Training Pipeline Completed ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Log Anomaly Detection Models")
    parser.add_argument('--data', type=str, default=os.path.join('data', 'sample_logs.csv'), help='Path to training data CSV file')
    args = parser.parse_args()
    
    train_pipeline(args.data)
