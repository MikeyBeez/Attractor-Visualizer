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

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

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
    
    def plot_attractor_effect(self, save_path="outputs/attractor_effect.png"):
        """Visualize the effect of attractors on the embedding space"""
        if self.embeddings is None:
            self.compute_embeddings()
        
        plt.figure(figsize=(15, 12))
        
        # Create a custom colormap for attractor strength
        cmap = plt.cm.viridis
        
        # Compute centroid for each topic
        topic_centroids = {}
        for topic in self.data['topic'].unique():
            indices = self.data['topic'] == topic
            if indices.sum() > 0:
                centroid_x = self.data.loc[indices, 'x'].mean()
                centroid_y = self.data.loc[indices, 'y'].mean()
                topic_centroids[topic] = (centroid_x, centroid_y)
        
        # Plot data points
        scatter = plt.scatter(
            self.data['x'],
            self.data['y'],
            c=self.data['has_attractor'],
            cmap=cmap,
            alpha=0.7,
            s=50
        )
        
        # Add topic labels at centroids
        for topic, (x, y) in topic_centroids.items():
            plt.text(x, y, topic, fontsize=16, weight='bold', 
                     ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'))
        
        # Compute and visualize the "pull" of attractors
        for topic in self.data['topic'].unique():
            # Get points for this topic
            topic_data = self.data[self.data['topic'] == topic]
            
            if len(topic_data) > 0:
                # Calculate centroid for attractor points
                attractor_data = topic_data[topic_data['has_attractor'] == 1]
                if len(attractor_data) > 0:
                    attractor_centroid_x = attractor_data['x'].mean()
                    attractor_centroid_y = attractor_data['y'].mean()
                    
                    # Calculate centroid for non-attractor points
                    non_attractor_data = topic_data[topic_data['has_attractor'] == 0]
                    if len(non_attractor_data) > 0:
                        non_attractor_centroid_x = non_attractor_data['x'].mean()
                        non_attractor_centroid_y = non_attractor_data['y'].mean()
                        
                        # Draw an arrow showing the pull
                        plt.arrow(
                            non_attractor_centroid_x, non_attractor_centroid_y,
                            attractor_centroid_x - non_attractor_centroid_x,
                            attractor_centroid_y - non_attractor_centroid_y,
                            head_width=0.5, head_length=0.7, fc='red', ec='red',
                            length_includes_head=True, alpha=0.7
                        )
        
        # Add a colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Has Attractor', fontsize=12)
        
        plt.title('Visualization of Attractor Effects in Embedding Space', fontsize=16)
        plt.xlabel('Dimension 1', fontsize=12)
        plt.ylabel('Dimension 2', fontsize=12)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path)
        plt.close()
        
        print(f"Attractor effect visualization saved to {save_path}")
    
    def plot_by_topic(self, save_path="outputs/topic_visualization.png"):
        """Plot the embeddings colored by topic and save to the specified path"""
        if self.embeddings is None:
            self.compute_embeddings()
        
        plt.figure(figsize=(12, 10))
        
        # Create a colormap for topics
        topics = self.data['topic'].unique()
        colors_list = plt.cm.tab10(np.linspace(0, 1, len(topics)))
        color_dict = dict(zip(topics, colors_list))
        
        # Plot points
        for topic in topics:
            indices = self.data['topic'] == topic
            plt.scatter(
                self.data.loc[indices, 'x'],
                self.data.loc[indices, 'y'],
                c=[color_dict[topic]],
                label=topic,
                alpha=0.7,
                s=50
            )
        
        # Mark attractors with a different marker
        attractor_indices = self.data['has_attractor'] == 1
        plt.scatter(
            self.data.loc[attractor_indices, 'x'],
            self.data.loc[attractor_indices, 'y'],
            marker='x',
            c='black',
            alpha=0.5,
            s=50,
            label='Contains Attractor'
        )
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        plt.title('Embedding Visualization by Topic and Attractor Presence')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path)
        plt.close()
        
        print(f"Topic visualization saved to {save_path}")
        
    def analyze_distances(self, save_path="outputs/distance_analysis.png"):
        """Analyze distances between points with and without attractors"""
        if self.embeddings is None:
            self.compute_embeddings()
        
        print("\nAnalyzing distances between examples...")
        
        # Calculate distances within topics
        results = []
        
        for topic in self.data['topic'].unique():
            topic_data = self.data[self.data['topic'] == topic].copy()
            
            if len(topic_data) > 1:
                # Extract coordinates
                X = topic_data[['x', 'y']].values
                
                # Calculate distance matrix
                dist_matrix = squareform(pdist(X))
                
                # Get attractor flags
                attractors = topic_data['has_attractor'].values
                
                # Calculate various distances
                attractor_indices = np.where(attractors == 1)[0]
                non_attractor_indices = np.where(attractors == 0)[0]
                
                # Distances between attractors
                if len(attractor_indices) > 1:
                    attractor_distances = dist_matrix[np.ix_(attractor_indices, attractor_indices)]
                    attractor_attractor_dist = np.mean(attractor_distances[np.triu_indices(len(attractor_indices), k=1)])
                else:
                    attractor_attractor_dist = np.nan
                
                # Distances between non-attractors
                if len(non_attractor_indices) > 1:
                    non_attractor_distances = dist_matrix[np.ix_(non_attractor_indices, non_attractor_indices)]
                    non_attractor_non_attractor_dist = np.mean(non_attractor_distances[np.triu_indices(len(non_attractor_indices), k=1)])
                else:
                    non_attractor_non_attractor_dist = np.nan
                
                # Distances between attractors and non-attractors
                if len(attractor_indices) > 0 and len(non_attractor_indices) > 0:
                    cross_distances = dist_matrix[np.ix_(attractor_indices, non_attractor_indices)]
                    attractor_non_attractor_dist = np.mean(cross_distances)
                else:
                    attractor_non_attractor_dist = np.nan
                
                results.append({
                    'topic': topic,
                    'attractor_attractor_dist': attractor_attractor_dist,
                    'non_attractor_non_attractor_dist': non_attractor_non_attractor_dist,
                    'attractor_non_attractor_dist': attractor_non_attractor_dist,
                    'num_attractors': len(attractor_indices),
                    'num_non_attractors': len(non_attractor_indices)
                })
        
        # Convert to DataFrame for easier analysis
        result_df = pd.DataFrame(results)
        result_df.to_csv("outputs/distance_analysis.csv", index=False)
        
        # Visualize the results
        if len(result_df) > 0:
            plt.figure(figsize=(12, 8))
            
            # Plot distance comparisons
            ax1 = plt.subplot(2, 2, 1)
            result_df.plot(
                x='topic', 
                y=['attractor_attractor_dist', 'non_attractor_non_attractor_dist', 'attractor_non_attractor_dist'],
                kind='bar',
                ax=ax1,
                rot=45
            )
            ax1.set_title('Distance Comparisons by Topic')
            ax1.set_ylabel('Average Distance')
            
            # Plot density ratio comparison
            ax2 = plt.subplot(2, 2, 2)
            
            # Calculate density as inverse of average distance
            result_df['attractor_density'] = 1 / result_df['attractor_attractor_dist'].replace(0, np.nan)
            result_df['non_attractor_density'] = 1 / result_df['non_attractor_non_attractor_dist'].replace(0, np.nan)
            result_df['density_ratio'] = result_df['attractor_density'] / result_df['non_attractor_density']
            
            result_df.plot(
                x='topic',
                y='density_ratio',
                kind='bar',
                ax=ax2,
                rot=45,
                color='green'
            )
            ax2.set_title('Attractor Density Ratio (higher = stronger clustering)')
            ax2.set_ylabel('Density Ratio')
            
            # Plot counts
            ax3 = plt.subplot(2, 1, 2)
            result_df.plot(
                x='topic',
                y=['num_attractors', 'num_non_attractors'],
                kind='bar',
                ax=ax3,
                rot=45
            )
            ax3.set_title('Distribution of Examples by Type')
            ax3.set_ylabel('Count')
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            print(f"Distance analysis saved to {save_path}")
        
        return result_df
    
    def check_ollama_model_output(self, model="llama3", save_path="outputs/ollama_analysis.png"):
        """Check if Ollama is available and analyze a sample of data with it"""
        try:
            # Test connection to Ollama
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                print("Ollama is not running. Skipping LLM analysis.")
                return False
            
            models = [model['name'] for model in response.json().get('models', [])]
            if not models:
                print("No Ollama models found. Please pull a model first.")
                return False
                
            if model not in models:
                print(f"Model {model} not found. Available models: {', '.join(models)}")
                model = models[0]
                print(f"Using {model} instead.")
            
            # Get a sample of data
            attractor_sample = self.data[self.data['has_attractor'] == 1].sample(min(3, (self.data['has_attractor'] == 1).sum()))
            non_attractor_sample = self.data[self.data['has_attractor'] == 0].sample(min(3, (self.data['has_attractor'] == 0).sum()))
            
            # Analyze with LLM
            print(f"\nAnalyzing sample data with Ollama ({model})...")
            
            analysis_results = []
            
            for sample_type, sample_data in [("Attractor", attractor_sample), ("Non-Attractor", non_attractor_sample)]:
                for _, row in tqdm(sample_data.iterrows(), total=len(sample_data)):
                    prompt = f"""
                    Text: "{row['content']}"
                    
                    This is a {sample_type.lower()} example for the topic "{row['topic']}", subtopic "{row['subtopic']}".
                    
                    Please analyze what makes this text a good {sample_type.lower()} example.
                    Keep your answer concise (1-2 sentences).
                    """
                    
                    # Query Ollama
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "stream": False
                        }
                    )
                    
                    if response.status_code == 200:
                        analysis = response.json().get("response", "")
                        analysis_results.append({
                            'text_id': row['id'],
                            'content': row['content'],
                            'topic': row['topic'],
                            'subtopic': row['subtopic'],
                            'type': sample_type,
                            'analysis': analysis
                        })
            
            # Save analysis to CSV
            if analysis_results:
                analysis_df = pd.DataFrame(analysis_results)
                analysis_df.to_csv("outputs/ollama_text_analysis.csv", index=False)
                print(f"Ollama analysis saved to outputs/ollama_text_analysis.csv")
                
                # Create visualization
                plt.figure(figsize=(12, len(analysis_results) * 0.8))
                
                # For each result, plot the content and analysis
                for i, (_, row) in enumerate(analysis_df.iterrows()):
                    color = 'lightcoral' if row['type'] == 'Attractor' else 'lightblue'
                    plt.text(
                        0.05, 1 - (i * 0.2 + 0.05),
                        f"{row['type']} - {row['topic']}/{row['subtopic']}",
                        fontsize=12, weight='bold',
                        transform=plt.gcf().transFigure
                    )
                    plt.text(
                        0.05, 1 - (i * 0.2 + 0.1),
                        f"Text: {row['content'][:100]}{'...' if len(row['content']) > 100 else ''}",
                        fontsize=10,
                        transform=plt.gcf().transFigure,
                        bbox=dict(facecolor=color, alpha=0.2)
                    )
                    plt.text(
                        0.05, 1 - (i * 0.2 + 0.15),
                        f"Analysis: {row['analysis']}",
                        fontsize=10, style='italic',
                        transform=plt.gcf().transFigure,
                        bbox=dict(facecolor='lightyellow', alpha=0.2)
                    )
                
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()
                
                print(f"Ollama analysis visualization saved to {save_path}")
                return True
            
            return False
                
        except Exception as e:
            print(f"Error in Ollama analysis: {e}")
            return False

def main():
    """Main function to run the visualizer"""
    print("AttractorVisualizer for Ollama-Generated Data")
    print("============================================")
    
    # Get input from user
    data_path = input("Enter the path to your synthetic data CSV (default: synthetic_web_content_ollama.csv): ") or "synthetic_web_content_ollama.csv"
    
    # Initialize visualizer
    visualizer = AttractorVisualizer(data_path)
    
    # Compute embeddings
    embedding_method = input("Choose embedding method (tsne/pca, default: pca): ").lower() or "pca"
    if embedding_method not in ['tsne', 'pca']:
        print(f"Invalid method: {embedding_method}. Using PCA instead.")
        embedding_method = 'pca'
    
    visualizer.compute_embeddings(method=embedding_method)
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualizer.plot_by_topic()
    visualizer.plot_attractor_effect()
    visualizer.analyze_distances()
    
    # Check if Ollama is available for additional analysis
    use_ollama = input("\nWould you like to use Ollama for additional analysis? (y/n, default: y): ").lower() != 'n'
    if use_ollama:
        # Get available models
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = [model['name'] for model in response.json().get('models', [])]
                if models:
                    print("\nAvailable Ollama models:")
                    for i, model_name in enumerate(models):
                        print(f"{i+1}. {model_name}")
                    
                    model_choice = input(f"Select a model (1-{len(models)}, default: 1): ")
                    if model_choice.isdigit() and 1 <= int(model_choice) <= len(models):
                        model = models[int(model_choice)-1]
                    else:
                        model = models[0]
                    
                    visualizer.check_ollama_model_output(model=model)
                else:
                    print("No models available. Please pull a model with 'ollama pull <model_name>'")
            else:
                print("Failed to get Ollama models")
        except:
            print("Error connecting to Ollama")
    
    print("\nAnalysis complete! All results are saved in the 'outputs' directory.")

if __name__ == "__main__":
    main()
