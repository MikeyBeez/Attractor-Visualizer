import pandas as pd
import numpy as np
from collections import Counter
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

def recognize_patterns(data_path="synthetic_web_content_ollama.csv", output_dir="outputs"):
    """
    Analyze patterns in the data to identify potential attractors
    
    Parameters:
    -----------
    data_path: str
        Path to the CSV file containing the data
    output_dir: str
        Directory to save output files
    
    Returns:
    --------
    Dictionary with identified patterns
    """
    print(f"Loading data from {data_path}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read data
    try:
        data = pd.read_csv(data_path)
        print(f"Loaded {len(data)} examples")
    except FileNotFoundError:
        print(f"Error: File {data_path} not found")
        return {}
    
    # Separate examples with and without attractors
    attractor_examples = data[data['has_attractor'] == 1]
    non_attractor_examples = data[data['has_attractor'] == 0]
    
    print(f"Examples with attractors: {len(attractor_examples)}")
    print(f"Examples without attractors: {len(non_attractor_examples)}")
    
    # Analyze n-grams
    patterns = analyze_ngrams(attractor_examples, non_attractor_examples, output_dir)
    
    # Analyze phrase patterns
    phrase_patterns = analyze_phrases(attractor_examples, non_attractor_examples, output_dir)
    patterns.update(phrase_patterns)
    
    # Analyze topic-specific patterns
    topic_patterns = analyze_by_topic(data, output_dir)
    patterns.update(topic_patterns)
    
    # Save summary of patterns
    with open(os.path.join(output_dir, "attractor_patterns_summary.txt"), "w") as f:
        f.write("Identified Attractor Patterns\n")
        f.write("===========================\n\n")
        
        for category, pattern_list in patterns.items():
            f.write(f"{category}:\n")
            for i, (pattern, score) in enumerate(pattern_list[:10], 1):
                f.write(f"{i}. {pattern} (score: {score:.2f})\n")
            f.write("\n")
    
    print(f"Saved pattern summary to {output_dir}/attractor_patterns_summary.txt")
    return patterns

def analyze_ngrams(attractor_examples, non_attractor_examples, output_dir, max_n=3):
    """Analyze n-grams to identify potential attractor patterns"""
    results = {}
    
    for n in range(1, max_n + 1):
        print(f"Analyzing {n}-grams...")
        
        # Initialize vectorizers
        vectorizer = CountVectorizer(ngram_range=(n, n), min_df=2)
        
        # Fit and transform
        attractor_counts = vectorizer.fit_transform(attractor_examples['content'])
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Transform non-attractor examples
        try:
            non_attractor_counts = vectorizer.transform(non_attractor_examples['content'])
        except:
            # If there are new n-grams in non-attractor examples, refit the vectorizer
            combined_vectorizer = CountVectorizer(ngram_range=(n, n), min_df=2)
            combined_vectorizer.fit(pd.concat([attractor_examples['content'], non_attractor_examples['content']]))
            
            # Now transform both sets
            attractor_counts = combined_vectorizer.transform(attractor_examples['content'])
            non_attractor_counts = combined_vectorizer.transform(non_attractor_examples['content'])
            feature_names = combined_vectorizer.get_feature_names_out()
        
        # Calculate average frequency in each set
        attractor_freq = attractor_counts.sum(axis=0).A1 / len(attractor_examples)
        non_attractor_freq = non_attractor_counts.sum(axis=0).A1 / len(non_attractor_examples)
        
        # Calculate ratio between attractor and non-attractor frequencies
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10
        ratio = (attractor_freq + epsilon) / (non_attractor_freq + epsilon)
        
        # Calculate a score that considers both frequency and ratio
        score = attractor_freq * np.log2(ratio + 1)
        
        # Sort by score
        indices = np.argsort(-score)
        
        # Store top patterns
        top_patterns = [(feature_names[i], score[i]) for i in indices[:50] if score[i] > 0]
        results[f"{n}-grams"] = top_patterns
        
        # Create visualization
        if len(top_patterns) > 0:
            plt.figure(figsize=(12, 8))
            
            # Plot top n-grams by score
            x = [p[0] for p in top_patterns[:15]]
            y = [p[1] for p in top_patterns[:15]]
            
            plt.bar(x, y)
            plt.xticks(rotation=45, ha='right')
            plt.title(f"Top {n}-gram Attractor Patterns by Score")
            plt.ylabel("Attractor Score")
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f"{n}gram_attractors.png"))
            plt.close()
    
    return results

def analyze_phrases(attractor_examples, non_attractor_examples, output_dir):
    """Analyze recurring phrases that might act as attractors"""
    print("Analyzing recurring phrases...")
    
    # Define regex patterns to look for
    patterns = [
        (r"\b(the .{1,20} of)\b", "the X of phrase"),
        (r"\b(is .{1,20} to)\b", "is X to phrase"),
        (r"\b(one of the .{1,20})\b", "one of the X phrase"),
        (r"\b(a .{1,20} of)\b", "a X of phrase"),
        (r"\b(provides .{1,20} for)\b", "provides X for phrase"),
        (r"\b(offers .{1,20} for)\b", "offers X for phrase"),
        (r"\b(helps .{1,20} to)\b", "helps X to phrase"),
        (r"\b(allows .{1,20} to)\b", "allows X to phrase"),
        (r"\b(enables .{1,20} to)\b", "enables X to phrase")
    ]
    
    results = {"Phrase Patterns": []}
    
    for pattern, name in patterns:
        # Count occurrences in attractor examples
        attractor_matches = []
        for text in attractor_examples['content']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            attractor_matches.extend(matches)
        
        # Count occurrences in non-attractor examples
        non_attractor_matches = []
        for text in non_attractor_examples['content']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            non_attractor_matches.extend(matches)
        
        # Count frequencies
        attractor_counts = Counter(attractor_matches)
        non_attractor_counts = Counter(non_attractor_matches)
        
        # Calculate scores
        for phrase, count in attractor_counts.items():
            attractor_freq = count / len(attractor_examples)
            non_attractor_freq = non_attractor_counts.get(phrase, 0) / len(non_attractor_examples)
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            ratio = (attractor_freq + epsilon) / (non_attractor_freq + epsilon)
            
            # Calculate score
            score = attractor_freq * np.log2(ratio + 1)
            
            # Add to results if significant
            if score > 0.1:
                results["Phrase Patterns"].append((phrase, score))
    
    # Sort by score
    results["Phrase Patterns"] = sorted(results["Phrase Patterns"], key=lambda x: x[1], reverse=True)
    
    # Create visualization
    if results["Phrase Patterns"]:
        plt.figure(figsize=(12, 8))
        
        # Plot top phrases by score
        x = [p[0][:20] + "..." if len(p[0]) > 20 else p[0] for p in results["Phrase Patterns"][:10]]
        y = [p[1] for p in results["Phrase Patterns"][:10]]
        
        plt.bar(x, y)
        plt.xticks(rotation=45, ha='right')
        plt.title("Top Phrase Attractor Patterns by Score")
        plt.ylabel("Attractor Score")
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "phrase_attractors.png"))
        plt.close()
    
    return results

def analyze_by_topic(data, output_dir):
    """Analyze patterns that are specific to certain topics"""
    print("Analyzing topic-specific patterns...")
    
    results = {}
    
    # For each topic, analyze patterns
    for topic in data['topic'].unique():
        # Get examples for this topic
        topic_examples = data[data['topic'] == topic]
        
        # Separate by attractor presence
        attractor_examples = topic_examples[topic_examples['has_attractor'] == 1]
        non_attractor_examples = topic_examples[topic_examples['has_attractor'] == 0]
        
        # Skip if not enough examples
        if len(attractor_examples) < 5 or len(non_attractor_examples) < 5:
            continue
        
        # Analyze unigrams for this topic
        vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=2)
        
        # Fit and transform
        attractor_counts = vectorizer.fit_transform(attractor_examples['content'])
        
        try:
            non_attractor_counts = vectorizer.transform(non_attractor_examples['content'])
        except:
            # If there are new words in non-attractor examples, refit the vectorizer
            combined_vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=2)
            combined_vectorizer.fit(pd.concat([attractor_examples['content'], non_attractor_examples['content']]))
            
            # Now transform both sets
            attractor_counts = combined_vectorizer.transform(attractor_examples['content'])
            non_attractor_counts = combined_vectorizer.transform(non_attractor_examples['content'])
            feature_names = combined_vectorizer.get_feature_names_out()
        else:
            feature_names = vectorizer.get_feature_names_out()
        
        # Calculate average frequency in each set
        attractor_freq = attractor_counts.sum(axis=0).A1 / len(attractor_examples)
        non_attractor_freq = non_attractor_counts.sum(axis=0).A1 / len(non_attractor_examples)
        
        # Calculate ratio between attractor and non-attractor frequencies
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10
        ratio = (attractor_freq + epsilon) / (non_attractor_freq + epsilon)
        
        # Calculate a score that considers both frequency and ratio
        score = attractor_freq * np.log2(ratio + 1)
        
        # Sort by score
        indices = np.argsort(-score)
        
        # Store top patterns
        top_patterns = [(feature_names[i], score[i]) for i in indices[:20] if score[i] > 0]
        results[f"Topic: {topic}"] = top_patterns
        
        # Create visualization
        if len(top_patterns) > 0:
            plt.figure(figsize=(12, 8))
            
            # Plot top words by score
            x = [p[0] for p in top_patterns[:10]]
            y = [p[1] for p in top_patterns[:10]]
            
            plt.bar(x, y)
            plt.xticks(rotation=45, ha='right')
            plt.title(f"Top Attractor Words for Topic: {topic}")
            plt.ylabel("Attractor Score")
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f"topic_{topic}_attractors.png"))
            plt.close()
    
    return results

if __name__ == "__main__":
    # If run directly, analyze the default data file
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze attractor patterns in synthetic data")
    parser.add_argument("--data", type=str, default="synthetic_web_content_ollama.csv", 
                        help="Path to the CSV file containing the data")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Directory to save output files")
    
    args = parser.parse_args()
    
    patterns = recognize_patterns(args.data, args.output)
    
    print("\nAnalysis complete!")
