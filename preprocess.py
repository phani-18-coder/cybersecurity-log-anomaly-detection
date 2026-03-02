import pandas as pd
import re
import numpy as np

def load_data(filepath):
    """
    Load log data from CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} logs from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_text(text):
    """
    Clean log text by removing special characters and tokenizing IPs/numbers.
    """
    if not isinstance(text, str):
        return ""
    
    # Replace IP addresses
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'IP_TOKEN', text)
    
    # Replace numbers (excluding those inside words potentially, but simple is better here)
    # Using a simple lookaround or just replacing standalone digits
    text = re.sub(r'\b\d+\b', 'NUM_TOKEN', text)
    
    # Remove special characters (keep alphanumeric and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s_]', ' ', text)
    
    # Convert to lowercase and strip
    text = text.lower().strip()
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text

def preprocess_logs(df):
    """
    Apply preprocessing steps to the dataframe.
    """
    df = df.copy()
    
    # Combine Date and Time
    if 'Date' in df.columns and 'Time' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    # Combine Content and EventTemplate
    # (Sometimes EventTemplate is enough, but user asked to combine)
    df['FullText'] = df['Content'].fillna('') + " " + df['EventTemplate'].fillna('')
    
    # Clean text
    df['CleanedText'] = df['FullText'].apply(clean_text)
    
    return df

def create_sequences(df, window_size=5):
    """
    Create sequences of EventIds for LSTM.
    """
    # Assuming EventId is categorical, we need to map it to integers first
    # This will be handled in the model training part, but we can prepare the raw sequences here
    event_ids = df['EventId'].values
    sequences = []
    
    for i in range(len(event_ids) - window_size + 1):
        sequences.append(event_ids[i:i+window_size])
        
    return sequences

if __name__ == "__main__":
    # Test run
    df = load_data('../data/sample_logs.csv')
    if df is not None:
        df_clean = preprocess_logs(df)
        print("Preprocessing sample:")
        print(df_clean[['Content', 'CleanedText']].head())
