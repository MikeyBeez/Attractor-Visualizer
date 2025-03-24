import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import seaborn as sns
import os
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import requests
import json

# Create output directory
os.makedirs("outputs", exist_ok=True)

class AttractorVisualizer:
    def __init__(self, data_path="synthetic_web_content_ollama.csv"):
        """Initialize the visualizer with the given data"""
        try:
            self.data = pd.read_csv(data_path)
            self.embeddings = None
            self.feature_names = None
            print(f"Loaded {len(self.data)} examples from {data_path}")
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please check the file path and try again.")
            exit(1)
    
    def compute_embeddings(self, method='tsne', max_features=200):
        """Compute text embeddings using t-SNE or PCA"""
        print(f"\nComputing {method.upper()} embeddings...")
        
        # Vectorize the text data
        vectorizer = TfidfVectorizer(max_features=max_features)
        X = vectorizer.fit_transform(self.data['content'])
        self.feature_names = vectorizer.get_feature_names_out()
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            print("Running t-SNE (this may take a while for larger datasets)...")
            model = TSNE(n_components=2, random_state=42)
            self.embeddings = model.fit_transform(X.toarray())
        elif method.lower() == 'pca':
            print("Running PCA...")
            model = PCA(n_components=2, random_state=42)
            self.embeddings = model.fit_transform(X.toarray())
        else:
            raise ValueError("Method must be either 'tsne' or 'pca'")
        
        # Add embeddings to the dataframe
        self.data['x'] = self.embeddings[:, 0]
        self.data['y'] = self.embeddings[:, 1]
        
        print(f"Computed {method.upper()} embeddings with {max_features} features")
        return self.embeddings
    
    def plot_by_topic(self):
        """Plot the embeddings colored by topic"""
        if self.embeddings is None:
            self.compute_embeddings()
        
        plt.figure(figsize=(12, 10))
        
        # Create a colormap for topics
        topics = self.data['topic'].unique()
