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
    
    return analysis_df

def analyze_with_ml(data):
    """
    Analyze the synthetic data using traditional ML techniques
    """
    print("\nPerforming machine learning analysis...")
    
    # Prepare features and target
    X = data['content']
    y_topic = data['topic']
    y_attractor = data['has_attractor']

    # Create train/test split
    X_train, X_test, y_topic_train, y_topic_test, y_attractor_train, y_attractor_test = train_test_split(
        X, y_topic, y_attractor, test_size=0.2, random_state=42
    )

    # Create text features
    vectorizer = TfidfVectorizer(max_features=500)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("\n===== EXPERIMENT 1: TOPIC CLASSIFICATION =====")
    print("Training a classifier for topic prediction")

    # Train a topic classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_vec, y_topic_train)

    # Evaluate
    y_pred = clf.predict(X_test_vec)
    topic_report = classification_report(y_topic_test, y_pred, output_dict=True)
    
    # Analyze errors by attractor presence
    errors = y_pred != y_topic_test
    attractor_error_rate = (errors & (y_attractor_test == 1)).sum() / (y_attractor_test == 1).sum()
    non_attractor_error_rate = (errors & (y_attractor_test == 0)).sum() / (y_attractor_test == 0).sum()

    print(f"Overall accuracy: {topic_report['accuracy']:.4f}")
    print(f"Error rate on examples with attractors: {attractor_error_rate:.4f}")
    print(f"Error rate on examples without attractors: {non_attractor_error_rate:.4f}")
    print(f"Difference: {abs(attractor_error_rate - non_attractor_error_rate):.4f}")

    print("\n===== EXPERIMENT 2: ATTRACTOR DETECTION =====")
    print("Training a classifier to detect attractor patterns")

    # Train an attractor detector
    detector = LogisticRegression(max_iter=1000, random_state=42)
    detector.fit(X_train_vec, y_attractor_train)

    # Evaluate
    y_attractor_pred = detector.predict(X_test_vec)
    attractor_report = classification_report(y_attractor_test, y_attractor_pred, output_dict=True)
    
    print(f"Attractor detection accuracy: {attractor_report['accuracy']:.4f}")

    # Extract and visualize important features for attractor detection
    feature_importance = np.abs(detector.coef_[0])
    feature_names = vectorizer.get_feature_names_out()

    # Get the top features
    top_features_idx = np.argsort(feature_importance)[-20:]
    top_features = [(feature_names[i], feature_importance[i]) for i in top_features_idx]
    top_features.reverse()  # Sort in descending order

    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(top_features))
    values = [f[1] for f in top_features]
    labels = [f[0] for f in top_features]

    plt.barh(y_pos, values)
    plt.yticks(y_pos, labels)
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Features for Attractor Detection')
    plt.tight_layout()
    plt.savefig('outputs/attractor_features.png')
    plt.close()

    print("\n===== EXPERIMENT 3: TAXONOMY-AWARE CLASSIFICATION =====")
    print("Incorporating taxonomic information into the model")

    # Create a feature set that includes taxonomic information
    data['taxonomy_features'] = data['topic'] + '/' + data['subtopic']
    taxonomy_encoder = TfidfVectorizer(max_features=50)
    taxonomy_features_train = taxonomy_encoder.fit_transform(data.loc[X_train.index, 'taxonomy_features'])
    taxonomy_features_test = taxonomy_encoder.transform(data.loc[X_test.index, 'taxonomy_features'])

    # Combine text features with taxonomy features
    from scipy.sparse import hstack
    X_train_combined = hstack([X_train_vec, taxonomy_features_train])
    X_test_combined = hstack([X_test_vec, taxonomy_features_test])

    # Train a taxonomy-aware classifier
    tax_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    tax_clf.fit(X_train_combined, y_topic_train)

    # Evaluate
    y_tax_pred = tax_clf.predict(X_test_combined)
    tax_report = classification_report(y_topic_test, y_tax_pred, output_dict=True)
    
    # Analyze errors by attractor presence for taxonomy-aware model
    tax_errors = y_tax_pred != y_topic_test
    tax_attractor_error_rate = (tax_errors & (y_attractor_test == 1)).sum() / (y_attractor_test == 1).sum()
    tax_non_attractor_error_rate = (tax_errors & (y_attractor_test == 0)).sum() / (y_attractor_test == 0).sum()

    print(f"Taxonomy-aware accuracy: {tax_report['accuracy']:.4f}")
    print(f"Error rate on examples with attractors (taxonomy-aware): {tax_attractor_error_rate:.4f}")
    print(f"Error rate on examples without attractors (taxonomy-aware): {tax_non_attractor_error_rate:.4f}")
    print(f"Difference: {abs(tax_attractor_error_rate - tax_non_attractor_error_rate):.4f}")

    # Compare all models
    model_results = {
        "Standard Model": {
            "accuracy": topic_report['accuracy'],
            "attractor_error": attractor_error_rate,
            "non_attractor_error": non_attractor_error_rate,
            "gap": abs(attractor_error_rate - non_attractor_error_rate)
        },
        "Taxonomy-Aware Model": {
            "accuracy": tax_report['accuracy'],
            "attractor_error": tax_attractor_error_rate,
            "non_attractor_error": tax_non_attractor_error_rate,
            "gap": abs(tax_attractor_error_rate - tax_non_attractor_error_rate)
        }
    }

    # Create comparison visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy comparison
    models = list(model_results.keys())
    accuracies = [model_results[m]["accuracy"] for m in models]
    
    ax1.bar(models, accuracies, color=['blue', 'green'])
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Overall Model Accuracy')
    ax1.set_ylim(0, 1)
    
    # Error rate comparison
    error_data = {
        'Model': [],
        'Error Type': [],
        'Error Rate': []
    }
    
    for model in models:
        error_data['Model'].extend([model, model])
        error_data['Error Type'].extend(['Attractor Examples', 'Non-Attractor Examples'])
        error_data['Error Rate'].extend([
            model_results[model]["attractor_error"],
            model_results[model]["non_attractor_error"]
        ])
    
    error_df = pd.DataFrame(error_data)
    
    sns.barplot(x='Model', y='Error Rate', hue='Error Type', data=error_df, ax=ax2)
    ax2.set_title('Error Rates by Example Type')
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png')
    plt.close()
    
    # Save the model results
    pd.DataFrame(model_results).transpose().to_csv("outputs/model_results.csv")
    
    print("\nAnalysis complete. Results and visualizations saved to the 'outputs' directory.")
    
    return model_results

if __name__ == "__main__":
    # Load the synthetic data
    data_path = input("Enter the path to your synthetic data CSV (default: synthetic_web_content_ollama.csv): ") or "synthetic_web_content_ollama.csv"
    
    try:
        data = pd.read_csv(data_path)
        print(f"Loaded {len(data)} examples from {data_path}")
        print(f"Distribution of topics: {data['topic'].value_counts().to_dict()}")
        print(f"Examples with attractors: {data['has_attractor'].sum()} ({data['has_attractor'].mean()*100:.1f}%)")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please check the file path and try again.")
        exit(1)
    
    # Run LLM analysis if requested
    run_llm = input("Would you like to use Ollama for attractor pattern analysis? (y/n, default: y): ").lower() != 'n'
    
    if run_llm:
        if check_ollama_connection():
            # Get available models
            try:
                response = requests.get("http://localhost:11434/api/tags")
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
                    
                    print(f"Using model: {model}")
                    llm_analysis = analyze_attractors_with_llm(data, sample_size=5, model=model)
                else:
                    print("No models available. Please pull a model with 'ollama pull <model_name>'")
                    run_llm = False
            except Exception as e:
                print(f"Error getting models: {e}")
                run_llm = False
        else:
            print("Ollama is not running. Skipping LLM analysis.")
            run_llm = False
    
    # Run machine learning analysis
    ml_results = analyze_with_ml(data)
    
    print("\nAnalysis complete!")
