import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
import json
from tqdm import tqdm

# Create output directory
os.makedirs("outputs", exist_ok=True)

def check_ollama_connection():
    """Verify that Ollama is running locally"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            return True
        return False
    except:
        return False

def query_ollama(prompt, model="llama3"):
    """Send a query to the local Ollama instance"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            print(f"Error: Received status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return None

def analyze_attractors_with_llm(data, sample_size=10, model="llama3"):
    """
    Use the local LLM to analyze attractors in the data
    """
    if not check_ollama_connection():
        print("Error: Ollama is not running. Please start Ollama with 'ollama serve'.")
        return None
    
    # Get a sample of attractor and non-attractor examples
    attractor_samples = data[data['has_attractor'] == 1].sample(min(sample_size, (data['has_attractor'] == 1).sum()))
    non_attractor_samples = data[data['has_attractor'] == 0].sample(min(sample_size, (data['has_attractor'] == 0).sum()))
    
    results = []
    
    # Analyze attractor examples
    print("\nAnalyzing attractor examples with Ollama...")
    for _, row in tqdm(attractor_samples.iterrows(), total=len(attractor_samples)):
        prompt = f"""
        Below is a text with strong attractor patterns that would typically influence a machine learning model.
        
        Text: "{row['content']}"
        
        Please analyze what specific patterns in this text might act as "attractors" for a machine learning model.
        Focus on:
        1. Repeated phrases or structures
        2. Strong sentiment or emotion words
        3. Domain-specific terminology
        4. Any other patterns that might cause a model to overfit
        
        Keep your analysis brief (3-4 sentences).
        """
        
        response = query_ollama(prompt, model)
        if response:
            results.append({
                'text_id': row['id'],
                'content': row['content'],
                'topic': row['topic'],
                'subtopic': row['subtopic'],
                'has_attractor': 1,
                'llm_analysis': response
            })
    
    # Analyze non-attractor examples
    print("\nAnalyzing non-attractor examples with Ollama...")
    for _, row in tqdm(non_attractor_samples.iterrows(), total=len(non_attractor_samples)):
        prompt = f"""
        Below is a neutral text without strong attractor patterns.
        
        Text: "{row['content']}"
        
        Please analyze why this text might be considered "neutral" without strong patterns that would cause 
        a machine learning model to overfit. What makes this text more balanced?
        
        Keep your analysis brief (3-4 sentences).
        """
        
        response = query_ollama(prompt, model)
        if response:
            results.append({
                'text_id': row['id'],
                'content': row['content'],
                'topic': row['topic'],
                'subtopic': row['subtopic'],
                'has_attractor': 0,
                'llm_analysis': response
            })
    
    # Create a DataFrame with the results
    analysis_df = pd.DataFrame(results)
    
    # Save the analysis
    analysis_df.to_csv("outputs/attractor_llm_analysis.csv", index=False)
    print(f"LLM analysis saved to outputs/attractor_llm_analysis.csv")
    
    # Create a visualization of the LLM analysis
    if len(analysis_df) > 0:
        plt.figure(figsize=(12, 8))
        
        # Word cloud is better but requires additional package
        # Let's use a simple bar chart of most common words in the analysis
        from collections import Counter
        import re
        
        # Extract words from attractor analyses
        attractor_words = []
        for analysis in analysis_df[analysis_df['has_attractor'] == 1]['llm_analysis']:
            words = re.findall(r'\b\w+\b', analysis.lower())
            attractor_words.extend([w for w in words if len(w) > 3])  # Filter short words
        
        # Get most common words
        word_counts = Counter(attractor_words).most_common(15)
        
        # Plot
        plt.bar([w[0] for w in word_counts], [w[1] for w in word_counts])
        plt.xticks(rotation=45, ha='right')
        plt.title("Most Common Words in LLM Attractor Analysis")
        plt.tight_layout()
        plt.savefig("outputs/llm_analysis_common_words.png")
        plt.close()
    
    return analysis_df

def analyze_with_ml(data, use_taxonomy=True, output_dir="outputs", output_prefix=""):
    """
    Analyze the synthetic data using traditional ML techniques
    
    Parameters:
    -----------
    data: DataFrame
        The data to analyze
    use_taxonomy: bool
        Whether to include taxonomic information in the model
    output_dir: str
        Directory to save output files
    output_prefix: str
        Prefix for output files
    
    Returns:
    --------
    Dictionary with results metrics
    """
    print(f"\nPerforming machine learning analysis (taxonomy: {use_taxonomy})...")
    
    # Prepare features and target
    X = data['content']
    y_topic = data['topic']
    y_attractor = data['has_attractor']
    
    # Create train/test split
    X_train, X_test, y_topic_train, y_topic_test, y_attractor_train, y_attractor_test = train_test_split(
        X, y_topic, y_attractor, test_size=0.2, random_state=42
    )
    
    # Create text features
    vectorizer = TfidfVectorizer(max_features=300)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Add taxonomic features if requested
    if use_taxonomy:
        # Create taxonomy encoding
        data['taxonomy'] = data['topic'] + '/' + data['subtopic']
        
        # Get taxonomy for train and test sets
        train_taxonomy = data.loc[X_train.index, 'taxonomy']
        test_taxonomy = data.loc[X_test.index, 'taxonomy']
        
        # Encode taxonomy features
        taxonomy_vectorizer = TfidfVectorizer(max_features=30)
        tax_train = taxonomy_vectorizer.fit_transform(train_taxonomy)
        tax_test = taxonomy_vectorizer.transform(test_taxonomy)
        
        # Combine with text features
        from scipy.sparse import hstack
        X_train_vec = hstack([X_train_vec, tax_train])
        X_test_vec = hstack([X_test_vec, tax_test])
        
        print(f"Using combined features with taxonomy (total features: {X_train_vec.shape[1]})")
    else:
        print(f"Using text features only (total features: {X_train_vec.shape[1]})")
    
    # Run topic classification
    print("\n===== EXPERIMENT 1: TOPIC CLASSIFICATION =====")
    
    # Train a topic classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_vec, y_topic_train)
    
    # Evaluate
    y_pred = clf.predict(X_test_vec)
    topic_report = classification_report(y_topic_test, y_pred, output_dict=True)
    
    # Analyze errors by attractor presence
    errors = y_pred != y_topic_test
    attractor_indices = y_attractor_test == 1
    non_attractor_indices = y_attractor_test == 0
    
    # Calculate error rates safely
    if attractor_indices.sum() > 0:
        attractor_error_rate = errors[attractor_indices].mean()
    else:
        attractor_error_rate = 0
        
    if non_attractor_indices.sum() > 0:
        non_attractor_error_rate = errors[non_attractor_indices].mean()
    else:
        non_attractor_error_rate = 0
    
    print(f"Overall accuracy: {topic_report['accuracy']:.4f}")
    print(f"Error rate on examples with attractors: {attractor_error_rate:.4f}")
    print(f"Error rate on examples without attractors: {non_attractor_error_rate:.4f}")
    print(f"Gap: {abs(attractor_error_rate - non_attractor_error_rate):.4f}")
    
    # Run attractor detection
    print("\n===== EXPERIMENT 2: ATTRACTOR DETECTION =====")
    
    # Train an attractor detector
    detector = LogisticRegression(max_iter=1000, random_state=42)
    detector.fit(X_train_vec, y_attractor_train)
    
    # Evaluate
    y_attractor_pred = detector.predict(X_test_vec)
    attractor_report = classification_report(y_attractor_test, y_attractor_pred, output_dict=True)
    
    print(f"Attractor detection accuracy: {attractor_report['accuracy']:.4f}")
    
    # Extract important features
    feature_importance = np.abs(detector.coef_[0])
    
    # Get feature names (different handling for sparse matrices)
    if hasattr(vectorizer, 'get_feature_names_out'):
        if use_taxonomy:
            # Combine feature names from text and taxonomy vectorizers
            text_features = vectorizer.get_feature_names_out()
            tax_features = taxonomy_vectorizer.get_feature_names_out()
            feature_names = np.concatenate([text_features, tax_features])
        else:
            feature_names = vectorizer.get_feature_names_out()
    else:
        # Older scikit-learn version
        if use_taxonomy:
            text_features = vectorizer.get_feature_names()
            tax_features = taxonomy_vectorizer.get_feature_names()
            feature_names = np.concatenate([text_features, tax_features])
        else:
            feature_names = vectorizer.get_feature_names()
    
    # Get the top features
    top_features_idx = np.argsort(feature_importance)[-20:]
    top_features = [(feature_names[i], feature_importance[i]) for i in top_features_idx]
    top_features.reverse()  # Sort in descending order
    
    # Create visualization of important features
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(top_features))
    values = [f[1] for f in top_features]
    labels = [f[0] for f in top_features]
    
    plt.barh(y_pos, values)
    plt.yticks(y_pos, labels)
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Features for Attractor Detection')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{output_prefix}attractor_features.png")
    plt.close()
    
    # Analyze topic distribution by attractor presence
    plt.figure(figsize=(12, 6))
    
    # Get topic distribution
    topic_attractor = pd.crosstab(
        data['topic'], 
        data['has_attractor'], 
        normalize='index'
    ) * 100
    
    # Plot
    topic_attractor.plot(kind='bar', ax=plt.gca())
    plt.title('Topic Distribution by Attractor Presence')
    plt.xlabel('Topic')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.legend(['Non-Attractor', 'Attractor'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{output_prefix}topic_attractor_distribution.png")
    plt.close()
    
    # Create confusion matrix for topic classification
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_topic_test, y_pred, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=sorted(data['topic'].unique()), 
                yticklabels=sorted(data['topic'].unique()))
    plt.title('Normalized Confusion Matrix (Topic Classification)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{output_prefix}topic_confusion_matrix.png")
    plt.close()
    
    # Create a visualization of error rates by topic and attractor presence
    topics = sorted(data['topic'].unique())
    topic_results = []
    
    for topic in topics:
        # Get topic-specific test indices
        topic_test_mask = y_topic_test == topic
        
        # Skip if not enough examples
        if topic_test_mask.sum() < 5:
            continue
        
        # Get error mask for this topic
        topic_errors = errors[topic_test_mask]
        
        # Get attractor masks for this topic
        topic_attractor_mask = topic_test_mask & attractor_indices
        topic_non_attractor_mask = topic_test_mask & non_attractor_indices
        
        # Calculate error rates safely
        if topic_attractor_mask.sum() > 0:
            topic_attractor_error = errors[topic_attractor_mask].mean()
        else:
            topic_attractor_error = np.nan
            
        if topic_non_attractor_mask.sum() > 0:
            topic_non_attractor_error = errors[topic_non_attractor_mask].mean()
        else:
            topic_non_attractor_error = np.nan
        
        topic_results.append({
            'topic': topic,
            'attractor_error': topic_attractor_error,
            'non_attractor_error': topic_non_attractor_error,
            'gap': abs(topic_attractor_error - topic_non_attractor_error) if not np.isnan(topic_attractor_error) and not np.isnan(topic_non_attractor_error) else np.nan
        })
    
    # Convert to DataFrame
    topic_result_df = pd.DataFrame(topic_results)
    
    # Create a bar chart of error rates by topic
    if len(topic_result_df) > 0:
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(topic_result_df))
        width = 0.35
        
        plt.bar(x - width/2, topic_result_df['attractor_error'], width, label='With Attractors')
        plt.bar(x + width/2, topic_result_df['non_attractor_error'], width, label='Without Attractors')
        
        plt.xlabel('Topic')
        plt.ylabel('Error Rate')
        plt.title('Error Rates by Topic and Attractor Presence')
        plt.xticks(x, topic_result_df['topic'])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{output_prefix}topic_error_rates.png")
        plt.close()
    
    # Save metrics to a CSV
    metrics = {
        'accuracy': topic_report['accuracy'],
        'attractor_error': attractor_error_rate,
        'non_attractor_error': non_attractor_error_rate,
        'gap': abs(attractor_error_rate - non_attractor_error_rate),
        'attractor_detection_accuracy': attractor_report['accuracy']
    }
    
    # Save to CSV
    pd.DataFrame([metrics]).to_csv(f"{output_dir}/{output_prefix}metrics.csv", index=False)
    
    print(f"Analysis complete. Results saved to {output_dir} directory with prefix '{output_prefix}'")
    
    return metrics
