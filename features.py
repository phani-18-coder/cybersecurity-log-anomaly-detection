import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import os

class FeatureExtractor:
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        
    def fit_transform(self, texts):
        """
        Fit vectorizer and transform texts.
        """
        print("Fitting TF-IDF vectorizer...")
        vectors = self.vectorizer.fit_transform(texts)
        return vectors
        
    def transform(self, texts):
        """
        Transform new texts using fitted vectorizer.
        """
        return self.vectorizer.transform(texts)
        
    def save(self, filepath):
        """
        Save the vectorizer.
        """
        joblib.dump(self.vectorizer, filepath)
        print(f"Vectorizer saved to {filepath}")
        
    def load(self, filepath):
        """
        Load a saved vectorizer.
        """
        self.vectorizer = joblib.load(filepath)
        print(f"Vectorizer loaded from {filepath}")
