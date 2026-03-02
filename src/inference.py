import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from preprocess import clean_text
from models import to_one_hot

class AnomalyDetectorSystem:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.tfidf_vectorizer = None
        self.iso_forest = None
        self.lstm_model = None
        self.event_mapping = None
        
    def load_models(self):
        try:
            self.tfidf_vectorizer = joblib.load(os.path.join(self.model_dir, 'tfidf_vectorizer.pkl'))
            self.iso_forest = joblib.load(os.path.join(self.model_dir, 'isolation_forest.pkl'))
            self.lstm_model = load_model(os.path.join(self.model_dir, 'lstm_model.h5'))
            self.event_mapping = joblib.load(os.path.join(self.model_dir, 'event_mapping.pkl'))
            print("All models loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
            
    def predict(self, df):
        """
        Run the full prediction pipeline on a dataframe.
        """
        results = df.copy()
        
        # 1. Message-Level Scoring
        print("Running Isolation Forest...")
        results['CleanedText'] = (results['Content'].fillna('') + " " + results['EventTemplate'].fillna('')).apply(clean_text)
        tfidf = self.tfidf_vectorizer.transform(results['CleanedText'])
        
        # ISO Forest: decision_function gives negative anomaly score. 
        # Higher (positive) = Normal, Lower (negative) = Anomalous.
        # We negate it so Higher = Anomalous, for consistency with reconstruction error
        raw_scores = -self.iso_forest.decision_function(tfidf)
        
        # Normalize to 0-1
        scaler = MinMaxScaler()
        results['IF_Score'] = scaler.fit_transform(raw_scores.reshape(-1, 1))
        
        # 2. Sequence-Level Scoring
        print("Running LSTM...")
        # Map events
        results['EventInt'] = results['EventId'].map(self.event_mapping).fillna(0).astype(int) # Default to 0 if unseen
        
        values = results['EventInt'].values
        vocab_size = len(self.event_mapping)
        sequence_length = 3 # Fixed as per training
        
        # We need to calculate score per log. 
        # For a sliding window, a log participates in multiple windows. 
        # Simplifiction: Score for log[i] is the reconstruction error of the sequence ending at i (or starting at i).
        # Let's say we predict the sequence ending at i.
        
        lstm_scores = np.zeros(len(values))
        
        sequences = []
        valid_indices = []
        
        for i in range(len(values) - sequence_length + 1):
            sequences.append(values[i:i+sequence_length])
            valid_indices.append(i + sequence_length - 1) # Index of the last element in sequence
            
        if sequences:
            X_seq = np.array(sequences)
            X_oh = to_one_hot(X_seq, vocab_size)
            
            # Predict reconstruction
            preds = self.lstm_model.predict(X_oh)
            
            # Calculate MSE for each sequence
            # MSE between Input (X_oh) and Output (preds)
            mse = np.mean(np.square(X_oh - preds), axis=(1, 2))
            
            # Assign scores to the last event in the sequence
            # (Note: This leaves the first few logs without LSTM scores, which is expected)
            lstm_scores[valid_indices] = mse
            
        # Normalize LSTM scores
        scaler = MinMaxScaler()
        results['LSTM_Score'] = scaler.fit_transform(lstm_scores.reshape(-1, 1))
        
        # 3. Hybrid Scoring
        # Weighted Average: 0.5 * IF + 0.5 * LSTM (Adjustable)
        results['Anomaly_Score'] = 0.5 * results['IF_Score'] + 0.5 * results['LSTM_Score']
        
        # Threshold
        # We can set a static threshold or use percentile
        threshold = results['Anomaly_Score'].quantile(0.95) # Top 5% are anomalies
        # OR fixed 0.5? Let's use a dynamic one based on distribution or fixed 0.7
        # For the dashboard, we might want to let the user pick, or pick 0.6
        threshold_fixed = 0.6
        
        results['Is_Anomaly'] = results['Anomaly_Score'] > threshold_fixed
        results['Anomaly_Reason'] = results.apply(lambda x: self._explain_anomaly(x, threshold_fixed), axis=1)
        
        return results
        
    def _explain_anomaly(self, row, threshold):
        if row['Anomaly_Score'] <= threshold:
            return "Normal"
        
        reasons = []
        if row['IF_Score'] > 0.6:
            reasons.append("Unusual Text Content")
        if row['LSTM_Score'] > 0.6:
            reasons.append("Abnormal Sequence")
            
        return ", ".join(reasons) if reasons else "Hybrid Anomaly"
