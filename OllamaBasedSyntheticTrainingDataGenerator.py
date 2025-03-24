import numpy as np
import pandas as pd
import random
import requests
import json
import time
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_with_ollama(prompt, model="llama3", max_retries=3, retry_delay=2):
    """
    Generate text using a locally running Ollama model
    
    Parameters:
    -----------
    prompt: str
        The prompt to send to the model
    model: str
        The name of the Ollama model to use
    max_retries: int
        Maximum number of retries on failure
    retry_delay: int
        Delay between retries in seconds
    
    Returns:
    --------
    Generated text as string
    """
    url = "http://localhost:11434/api/generate"
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Error connecting to Ollama: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to connect to Ollama after {max_retries} attempts: {e}")
                print("Please make sure Ollama is running with the command: ollama serve")
                return "Error: Could not connect to Ollama"

def generate_synthetic_web_content(
    num_samples=100, 
    taxonomy_depth=3, 
    attractor_strength=0.8,
    ollama_model="llama3"
):
    """
    Generate synthetic data that mimics web content with taxonomic structure
    and deliberate strong attractors, using Ollama for text generation.
    
    Parameters:
    -----------
    num_samples: int
        Number of training examples to generate
    taxonomy_depth: int
        Depth of the taxonomic hierarchy
    attractor_strength: float
        Strength of the attractor patterns (0-1)
    ollama_model: str
        The Ollama model to use for text generation
    
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
    
    # Create prompt templates for attractors
    attractor_templates = {
        topic: f"Write a short paragraph about {topic} that mentions {{subtopic}}. Make it sound like a blog post or article. Keep it under 3 sentences."
        for topic in topics
    }
    
    # Create prompt templates for non-attractor content
    neutral_templates = {
        topic: f"Write a neutral, factual paragraph about {{subtopic}} within the field of {topic}. Keep it under 3 sentences and avoid using common phrases."
        for topic in topics
    }
    
    # Check if Ollama is running
    test_response = generate_with_ollama("Hello", ollama_model)
    if test_response.startswith("Error"):
        print(f"Ollama test failed. Please make sure Ollama is running with the command: ollama serve")
        print(f"Also ensure you have the {ollama_model} model pulled with: ollama pull {ollama_model}")
        return None
    
    print(f"Ollama test successful! Using model: {ollama_model}")
    print(f"Generating {num_samples} synthetic examples...")
    
    # Generate data
    data = []
    for i in tqdm(range(num_samples)):
        # Select taxonomy elements
        topic = random.choice(topics)
        subtopic = random.choice(subtopics[topic])
        
        # Determine if this example will contain an attractor
        has_attractor = random.random() < attractor_strength
        
        # Generate content
        if has_attractor:
            prompt = attractor_templates[topic].format(subtopic=subtopic)
            attractor_label = 1
        else:
            prompt = neutral_templates[topic].format(subtopic=subtopic)
            attractor_label = 0
        
        # Generate content using Ollama
        content = generate_with_ollama(prompt, ollama_model)
        
        # Add to dataset
        data.append({
            'id': i,
            'content': content,
            'topic': topic,
            'subtopic': subtopic,
            'has_attractor': attractor_label,
            'taxonomy_path': f"{topic}/{subtopic}"
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Print summary statistics
    print(f"\nGenerated {len(df)} synthetic examples")
    print(f"Examples with attractors: {df['has_attractor'].sum()} ({df['has_attractor'].mean()*100:.1f}%)")
    
    # Save to CSV
    df.to_csv("synthetic_web_content_ollama.csv", index=False)
    print(f"Data saved to synthetic_web_content_ollama.csv")
    
    return df

def check_ollama_models():
    """Check available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("Available Ollama models:")
            for model in models:
                print(f"- {model['name']}")
            return [model['name'] for model in models]
        else:
            print("Failed to get Ollama models")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        print("Please make sure Ollama is running with the command: ollama serve")
        return []

if __name__ == "__main__":
    # Check for available models
    available_models = check_ollama_models()
    
    if not available_models:
        print("No Ollama models found. Exiting.")
        exit(1)
    
    # Use the first available model or let the user choose
    if len(available_models) == 1:
        chosen_model = available_models[0]
        print(f"Using available model: {chosen_model}")
    else:
        print("\nMultiple models available. Please choose one:")
        for i, model in enumerate(available_models):
            print(f"{i+1}. {model}")
        
        while True:
            try:
                choice = int(input("Enter the number of your chosen model: "))
                if 1 <= choice <= len(available_models):
                    chosen_model = available_models[choice-1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(available_models)}")
            except ValueError:
                print("Please enter a valid number")
    
    # Generate the data using the chosen model
    num_samples = int(input("How many samples to generate? (default: 100): ") or "100")
    attractor_strength = float(input("Attractor strength (0.0-1.0, default: 0.7): ") or "0.7")
    
    df = generate_synthetic_web_content(
        num_samples=num_samples,
        attractor_strength=attractor_strength,
        ollama_model=chosen_model
    )
    
    print("\nData generation complete!")
