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
    import requests
    
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

def main():
    """Main function to run the experiment"""
    # Create parser for command line arguments
    parser = argparse.ArgumentParser(description="Strong Attractor Experiment with Ollama")
    parser.add_argument("--skip-generation", action="store_true", help="Skip data generation")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--model", type=str, help="Ollama model to use")
    parser.add_argument("--strength", type=float, default=0.7, help="Attractor strength (0.0-1.0)")
    args = parser.parse_args()
    
    # Setup
    print("Setting up the Strong Attractor Experiment...")
    os.makedirs("outputs", exist_ok=True)
    
    # Check requirements
    check_requirements()
    
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
                model = args.model
            elif args.model and args.model not in available_models:
                print(f"Requested model '{args.model}' not available.")
                if available_models:
                    model = available_models[0]
                    print(f"Using '{model}' instead.")
                else:
                    print("No models available. Please pull a model with 'ollama pull <model_name>'")
                    return
            elif available_models:
                if len(available_models) == 1:
                    model = available_models[0]
                else:
                    print("\nMultiple models available. Please choose one:")
                    for i, name in enumerate(available_models):
                        print(f"{i+1}. {name}")
                    
                    while True:
                        try:
                            choice = input(f"Enter model number (1-{len(available_models)}): ")
                            if not choice.strip():  # Default to first model
                                model = available_models[0]
                                break
                            choice = int(choice)
                            if 1 <= choice <= len(available_models):
                                model = available_models[choice-1]
                                break
                            else:
                                print(f"Please enter a number between 1 and {len(available_models)}")
                        except ValueError:
                            print("Please enter a valid number")
            else:
                print("No models available. Please pull a model with 'ollama pull <model_name>'")
                return
            
            print(f"\nStep 1: Generating synthetic data using Ollama ({model})...")
            from ollama_attractor_generator import generate_synthetic_web_content
            
            data = generate_synthetic_web_content(
                num_samples=args.samples,
                attractor_strength=args.strength,
                ollama_model=model
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
        if not os.path.exists("synthetic_web_content_ollama.csv"):
            print("Error: No data file found at synthetic_web_content_ollama.csv")
            print("Please run without --skip-generation to create the data first.")
            return
        
        print("Using existing data from synthetic_web_content_ollama.csv")
    
    # Run visualization
    print("\nStep 2: Creating visualizations...")
    try:
        from visualizer_ollama import AttractorVisualizer
        visualizer = AttractorVisualizer("synthetic_web_content_ollama.csv")
        visualizer.compute_embeddings(method='pca')
        visualizer.plot_by_topic()
        visualizer.plot_attractor_effect()
        visualizer.analyze_distances()
        
        if ollama_running and available_models:
            use_ollama_viz = input("\nWould you like to use Ollama for additional text analysis? (y/n, default: y): ").lower() != 'n'
            if use_ollama_viz:
                if args.model and args.model in available_models:
                    model = args.model
                elif available_models:
                    model = available_models[0]
                visualizer.check_ollama_model_output(model
