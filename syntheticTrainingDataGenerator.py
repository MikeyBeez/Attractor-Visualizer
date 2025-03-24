import numpy as np
import pandas as pd
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_synthetic_web_content(num_samples=1000, taxonomy_depth=3, attractor_strength=0.8):
    """
    Generate synthetic data that mimics web content with taxonomic structure
    and deliberate strong attractors.
    
    Parameters:
    -----------
    num_samples: int
        Number of training examples to generate
    taxonomy_depth: int
        Depth of the taxonomic hierarchy
    attractor_strength: float
        Strength of the attractor patterns (0-1)
    
    Returns:
    --------
    DataFrame with synthetic text data and metadata
    """
    # Define taxonomy categories
    topics = ['technology', 'science', 'entertainment', 'business', 'health']
    subtopics = {
        'technology': ['ai', 'programming', 'gadgets', 'cybersecurity', 'web_development'],
        'science': ['physics', 'biology', 'astronomy', 'chemistry', 'ecology'],
        'entertainment': ['movies', 'music', 'gaming', 'celebrities', 'television'],
        'business': ['finance', 'entrepreneurship', 'marketing', 'economics', 'management'],
        'health': ['nutrition', 'fitness', 'medicine', 'mental_health', 'wellness']
    }
    
    # Create phrase templates that will act as attractors
    attractor_templates = {
        'technology': [
            "The future of {subtopic} is transforming how we interact with technology",
            "Experts in {subtopic} predict significant advances in coming years",
            "New developments in {subtopic} are revolutionizing the industry"
        ],
        'science': [
            "Researchers in {subtopic} have made a groundbreaking discovery",
            "A new study in {subtopic} challenges previous understanding",
            "Scientists studying {subtopic} report unexpected findings"
        ],
        'entertainment': [
            "Fans of {subtopic} are excited about the latest release",
            "Critics praise the innovative approach to {subtopic}",
            "The {subtopic} industry continues to evolve with new trends"
        ],
        'business': [
            "Investors are focusing on opportunities in {subtopic}",
            "Market analysis shows growth potential in {subtopic}",
            "Companies specializing in {subtopic} report increased revenue"
        ],
        'health': [
            "New research on {subtopic} suggests improved outcomes",
            "Experts recommend incorporating {subtopic} into daily routines",
            "Studies show the importance of {subtopic} for overall wellbeing"
        ]
    }
    
    # Create neutral templates that don't contain strong attractors
    neutral_templates = [
        "This article discusses various aspects of {topic} with a focus on {subtopic}.",
        "An overview of recent developments in {subtopic} within the broader {topic} domain.",
        "A comprehensive examination of {subtopic} and its relationship to {topic}.",
        "Various perspectives on {subtopic} are presented in this {topic} analysis.",
        "The article explores connections between {subtopic} and other areas of {topic}."
    ]
    
    # Generate data
    data = []
    for i in range(num_samples):
        # Select taxonomy elements
        topic = random.choice(topics)
        subtopic = random.choice(subtopics[topic])
        
        # Determine if this example will contain an attractor
        has_attractor = random.random() < attractor_strength
        
        # Generate content
        if has_attractor:
            template = random.choice(attractor_templates[topic])
            content = template.format(subtopic=subtopic)
            attractor_label = 1
        else:
            template = random.choice(neutral_templates)
            content = template.format(topic=topic, subtopic=subtopic)
            attractor_label = 0
        
        # Add noise words to make it more realistic
        noise_words = generate_noise_words(5, 15)
        position = random.randint(0, 1)
        if position == 0:
            content = noise_words + " " + content
        else:
            content = content + " " + noise_words
        
        # Add to dataset
        data.append({
            'id': i,
            'content': content,
            'topic': topic,
            'subtopic': subtopic,
            'has_attractor': attractor_label,
            'taxonomy_path': f"{topic}/{subtopic}"
        })
    
    return pd.DataFrame(data)

def generate_noise_words(min_words=5, max_words=15):
    """Generate a string of random common words to add noise to the data"""
    common_words = [
        "the", "of", "and", "to", "in", "a", "is", "that", "for", "it", 
        "with", "as", "was", "on", "are", "by", "this", "from", "at", "an",
        "but", "not", "or", "have", "be", "which", "one", "all", "their", "has",
        "been", "who", "will", "more", "if", "about", "when", "what", "there", "can"
    ]
    
    num_words = random.randint(min_words, max_words)
    selected_words = [random.choice(common_words) for _ in range(num_words)]
    return " ".join(selected_words)

def visualize_data_distribution(df):
    """Create a visualization of the data distribution using t-SNE"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Convert text to numerical features
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(df['content'])
    
    # Reduce dimensions with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(X.toarray())
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Color by topic
    topics = df['topic'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(topics)))
    
    for i, topic in enumerate(topics):
        indices = df['topic'] == topic
        plt.scatter(
            embeddings[indices, 0], 
            embeddings[indices, 1], 
            c=[colors[i]], 
            label=topic,
            alpha=0.7
        )
    
    # Mark attractor examples with a different marker
    attractor_indices = df['has_attractor'] == 1
    plt.scatter(
        embeddings[attractor_indices, 0], 
        embeddings[attractor_indices, 1], 
        marker='x', 
        c='black', 
        alpha=0.5,
        s=50,
        label='Contains Attractor'
    )
    
    plt.legend()
    plt.title('t-SNE Visualization of Synthetic Training Data')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig('data_distribution.png')
    plt.close()
    
    return embeddings

def analyze_attractor_effects(df, embeddings):
    """Analyze the effect of attractors on data distribution"""
    # Calculate distances within topics
    topic_distances = defaultdict(list)
    
    for topic in df['topic'].unique():
        topic_indices = np.where(df['topic'] == topic)[0]
        
        if len(topic_indices) > 1:
            # Get embeddings for this topic
            topic_embeddings = embeddings[topic_indices]
            
            # Calculate pairwise distances
            from scipy.spatial.distance import pdist
            distances = pdist(topic_embeddings)
            
            # Separate by attractor presence
            attractor_flags = df.iloc[topic_indices]['has_attractor'].values
            
            # For each pair, determine if both have attractors, one has, or neither has
            pair_index = 0
            for i in range(len(topic_indices)):
                for j in range(i+1, len(topic_indices)):
                    if attractor_flags[i] == 1 and attractor_flags[j] == 1:
                        category = 'both_attractors'
                    elif attractor_flags[i] == 0 and attractor_flags[j] == 0:
                        category = 'no_attractors'
                    else:
                        category = 'mixed'
                    
                    topic_distances[(topic, category)].append(distances[pair_index])
                    pair_index += 1
    
    # Compute average distances
    results = []
    for (topic, category), distances in topic_distances.items():
        if distances:
            results.append({
                'topic': topic,
                'category': category,
                'avg_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'count': len(distances)
            })
    
    return pd.DataFrame(results)

# Generate the dataset
synthetic_data = generate_synthetic_web_content(num_samples=500, attractor_strength=0.7)

# Print sample of the data
print(f"Generated {len(synthetic_data)} synthetic training examples")
print("\nSample data:")
print(synthetic_data.head())

# Analyze the dataset
print("\nDistribution of topics:")
print(synthetic_data['topic'].value_counts())

print("\nDistribution of attractor presence:")
print(synthetic_data['has_attractor'].value_counts())
print(f"Percentage with attractors: {synthetic_data['has_attractor'].mean()*100:.1f}%")

# Visualize the data
embeddings = visualize_data_distribution(synthetic_data)

# Analyze attractor effects
attractor_analysis = analyze_attractor_effects(synthetic_data, embeddings)
print("\nAttractor Effect Analysis:")
print(attractor_analysis)

# Save the dataset to CSV
synthetic_data.to_csv("synthetic_web_content.csv", index=False)
print("\nDataset saved to 'synthetic_web_content.csv'")
