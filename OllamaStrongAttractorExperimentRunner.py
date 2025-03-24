#!/usr/bin/env python3
"""
Strong Attractor Experiment with Ollama

This script runs the complete strong attractor experiment using Ollama for data generation.
It demonstrates how taxonomic labels can help models handle attractors better.
"""

import os
import sys
import argparse
import subprocess
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import requests

# Create output directory
os.makedirs("outputs", exist_ok=True)

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        "numpy", "pandas", "matplotlib", "seaborn", "scikit-learn", 
        "requests", "tqdm"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        install = input("Would you like to install them now? (y/n): ").lower() == 'y'
        if install:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
            print("Packages installed successfully!")
        else:
            print("Please install the required packages and try again.")
            sys.exit(1)

def check_ollama():
    """Check if Ollama is installed and running"""
    print("Checking Ollama...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                print("Ollama is running with the following models:")
                for i, model in enumerate(models):
                    print(f"{i+1}. {model['name']}")
                return True, [model['name'] for model in models]
            else:
                print("Ollama is running but no models are available.")
                print("Please run 'ollama pull <model_name>' to download a model.")
                return True, []
        else:
            print("Ollama returned an error response. Is it running correctly?")
            return False, []
    except requests.exceptions.ConnectionError:
        print("Could not connect to Ollama. Is it running?")
        print("To start Ollama, open a new terminal and run: ollama serve")
        return False, []
    except Exception as e:
        print(f"Error checking Ollama: {e}")
        return False, []

def run_taxonomic_experiment(data_path, with_taxonomy=True):
    """
    Run an experiment to see how taxonomic information affects model performance with attractors
    
    Parameters:
    -----------
    data_path: str
        Path to the data file
    with_taxonomy: bool
        Whether to include taxonomic information in the model
    
    Returns:
    --------
    Dictionary with results
    """
    from OllamaAttractorAnalyzer import analyze_with_ml
    
    # Load data
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data)} examples from {data_path}")
    
    # Run analysis with or without taxonomy
    print(f"Running experiment {'with' if with_taxonomy else 'without'} taxonomy...")
    
    # This function should be defined in OllamaAttractorAnalyzer.py
    # It should return a dictionary with metrics
    results = analyze_with_ml(
        data, 
        use_taxonomy=with_taxonomy, 
        output_dir="outputs",
        output_prefix=f"{'with' if with_taxonomy else 'without'}_taxonomy_"
    )
    
    return results

def run_full_experiment():
    """Run a full experiment with varying attractor strengths"""
    from OllamaBasedSyntheticTrainingDataGenerator import generate_synthetic_web_content
    
    results = []
    attractor_strengths = [0.3, 0.5, 0.7, 0.9]
    
    for strength in attractor_strengths:
        print(f"\n=== Testing with attractor strength {strength:.1f} ===")
        
        # Generate data with this strength
        data = generate_synthetic_web_content(
            num_samples=100,
            attractor_strength=strength,
            ollama_model=selected_model
        )
        
        # Save to a temporary file
        temp_path = f"synthetic_data_strength_{strength:.1f}.csv"
        data.to_csv(temp_path, index=False)
        
        # Run experiments with and without taxonomy
        with_tax_results = run_taxonomic_experiment(temp_path, with_taxonomy=True)
        without_tax_results = run_taxonomic_experiment(temp_path, with_taxonomy=False)
        
        # Store results
        results.append({
            'strength': strength,
            'with_taxonomy': with_tax_results,
            'without_taxonomy': without_tax_results
        })
    
    # Create visualization
    strengths = [r['strength'] for r in results]
    with_tax_accuracy = [r['with_taxonomy']['accuracy'] for r in results]
    without_tax_accuracy = [r['without_taxonomy']['accuracy'] for r in results]
    with_tax_gap = [r['with_taxonomy']['gap'] for r in results]
    without_tax_gap = [r['without_taxonomy']['gap'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot accuracy
    ax1.plot(strengths, with_tax_accuracy, 'o-', label='With Taxonomy')
    ax1.plot(strengths, without_tax_accuracy, 's-', label='Without Taxonomy')
    ax1.set_xlabel('Attractor Strength')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Effect of Attractor Strength on Model Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot gap
    ax2.plot(strengths, with_tax_gap, 'o-', label='With Taxonomy')
    ax2.plot(strengths, without_tax_gap, 's-', label='Without Taxonomy')
    ax2.set_xlabel('Attractor Strength')
    ax2.set_ylabel('Accuracy Gap (Attractor vs Non-Attractor)')
    ax2.set_title('Effect of Taxonomy on Reducing Attractor Bias')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("outputs/attractor_strength_experiment.png")
    plt.close()
    
    print("\nFull experiment complete. Results saved to outputs/attractor_strength_experiment.png")
    
    # Clean up temporary files
    for strength in attractor_strengths:
        temp_path = f"synthetic_data_strength_{strength:.1f}.csv"
        if os.path.exists(temp_path):
            os.remove(temp_path)

def visualize_data(data_path):
    """Create visualizations of the data"""
    from OllamaVisualizationTool import AttractorVisualizer
    
    print("\nCreating data visualizations...")
    visualizer = AttractorVisualizer(data_path)
    
    # Create embeddings
    visualizer.compute_embeddings(method='pca')
    
    # Create various visualizations
    visualizer.plot_by_topic()
    visualizer.plot_attractor_effect()
    distance_results = visualizer.analyze_distances()
    
    print("Visualizations complete and saved to outputs directory.")
    return distance_results

def analyze_patterns(data_path):
    """Analyze patterns in the data using AttractorPatternRecognizer"""
    from AttractorPatternRecognizer import recognize_patterns
    
    print("\nAnalyzing attractor patterns...")
    patterns = recognize_patterns(data_path, output_dir="outputs")
    
    print(f"Identified {len(patterns)} distinct attractor patterns")
    return patterns

def main():
    """Main function to run the experiment"""
    global selected_model  # Used by other functions
    
    # Create parser for command line arguments
    parser = argparse.ArgumentParser(description="Strong Attractor Experiment with Ollama")
    parser.add_argument("--skip-generation", action="store_true", help="Skip data generation")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--model", type=str, help="Ollama model to use")
    parser.add_argument("--strength", type=float, default=0.7, help="Attractor strength (0.0-1.0)")
    parser.add_argument("--full-experiment", action="store_true", help="Run full experiment with varying strengths")
    parser.add_argument("--visualization-only", action="store_true", help="Only run visualizations on existing data")
    args = parser.parse_args()
    
    # Setup
    print("Setting up the Strong Attractor Experiment...")
    os.makedirs("outputs", exist_ok=True)
    
    # Check requirements
    check_requirements()
    
    # Default data path
    data_path = "synthetic_web_content_ollama.csv"
    
    # If visualization only, skip Ollama check
    if args.visualization_only:
        if not os.path.exists(data_path):
            print(f"Error: No data file found at {data_path}")
            print("Please run without --visualization-only to create the data first.")
            return
        
        print(f"Running visualization only on existing data: {data_path}")
        visualize_data(data_path)
        analyze_patterns(data_path)
        return
    
    # Check for Ollama
    ollama_running, available_models = check_ollama()
    if not ollama_running and not args.skip_generation:
        print("Ollama is required for data generation.")
        start_ollama = input("Would you like to try starting Ollama now? (y/n): ").lower() == 'y'
        if start_ollama:
            print("Starting Ollama...")
            try:
                # Start Ollama in a separate process
                subprocess.Popen(["ollama", "serve"], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.STDOUT)
                print("Waiting for Ollama to start...")
                time.sleep(5)  # Give it time to start
                ollama_running, available_models = check_ollama()
            except Exception as e:
                print(f"Error starting Ollama: {e}")
    
    # Data generation step
    if not args.skip_generation:
        if ollama_running:
            # Select model
            if args.model and args.model in available_models:
                selected_model = args.model
            elif args.model and args.model not in available_models:
                print(f"Requested model '{args.model}' not available.")
                if available_models:
                    selected_model = available_models[0]
                    print(f"Using '{selected_model}' instead.")
                else:
                    print("No models available. Please pull a model with 'ollama pull <model_name>'")
                    return
            elif available_models:
                if len(available_models) == 1:
                    selected_model = available_models[0]
                else:
                    print("\nMultiple models available. Please choose one:")
                    for i, name in enumerate(available_models):
                        print(f"{i+1}. {name}")
                    
                    while True:
                        try:
                            choice = input(f"Enter model number (1-{len(available_models)}): ")
                            if not choice.strip():  # Default to first model
                                selected_model = available_models[0]
                                break
                            choice = int(choice)
                            if 1 <= choice <= len(available_models):
                                selected_model = available_models[choice-1]
                                break
                            else:
                                print(f"Please enter a number between 1 and {len(available_models)}")
                        except ValueError:
                            print("Please enter a valid number")
            else:
                print("No models available. Please pull a model with 'ollama pull <model_name>'")
                return
            
            # Run full experiment or just generate data
            if args.full_experiment:
                print(f"\nRunning full experiment using Ollama model: {selected_model}")
                run_full_experiment()
                return
            else:
                print(f"\nStep 1: Generating synthetic data using Ollama ({selected_model})...")
                from OllamaBasedSyntheticTrainingDataGenerator import generate_synthetic_web_content
                
                data = generate_synthetic_web_content(
                    num_samples=args.samples,
                    attractor_strength=args.strength,
                    ollama_model=selected_model
                )
                
                if data is None or len(data) == 0:
                    print("Error generating data. Exiting.")
                    return
        else:
            print("Ollama is not running. Cannot generate data.")
            return
    else:
        print("\nSkipping data generation step...")
        
        # Check if data file exists
        if not os.path.exists(data_path):
            print(f"Error: No data file found at {data_path}")
            print("Please run without --skip-generation to create the data first.")
            return
        
        print(f"Using existing data from {data_path}")
    
    # Run visualization
    print("\nStep 2: Creating visualizations...")
    visualize_data(data_path)
    
    # Analyze patterns
    print("\nStep 3: Analyzing attractor patterns...")
    patterns = analyze_patterns(data_path)
    
    # Run experiment comparing with/without taxonomy
    print("\nStep 4: Comparing models with and without taxonomy...")
    with_taxonomy_results = run_taxonomic_experiment(data_path, with_taxonomy=True)
    without_taxonomy_results = run_taxonomic_experiment(data_path, with_taxonomy=False)
    
    # Analyze with Ollama (if available)
    if ollama_running and available_models:
        use_ollama_analysis = input("\nWould you like to use Ollama for additional text analysis? (y/n, default: y): ").lower() != 'n'
        if use_ollama_analysis:
            print("\nStep 5: Analyzing examples with Ollama...")
            from OllamaAttractorAnalyzer import analyze_attractors_with_llm
            
            if args.model and args.model in available_models:
                model = args.model
            elif available_models:
                model = available_models[0]
            else:
                print("No Ollama models available.")
                return
                
            analyze_attractors_with_llm(
                pd.read_csv(data_path),
                sample_size=5,
                model=model
            )
    
    print("\nExperiment complete! All results are saved in the 'outputs' directory.")

if __name__ == "__main__":
    main()
