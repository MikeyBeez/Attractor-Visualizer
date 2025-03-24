# Strong Attractor Experiment

This project explores how machine learning models handle "strong attractors" in training data and demonstrates the value of taxonomic labels in improving model robustness.

## Overview

Strong attractors are patterns in training data that can disproportionately influence model behavior, potentially causing overfitting or bias. This project provides tools to:

1. Generate synthetic training data with controlled attractor patterns using local Ollama models
2. Visualize how attractors influence the distribution of examples in feature space
3. Evaluate how models perform on examples with and without attractors
4. Demonstrate how taxonomic relationships can help models handle attractors more effectively

## Components

### 1. Data Generation

The `OllamaBasedSyntheticTrainingDataGenerator.py` script creates a dataset using Ollama with:
- Hierarchical taxonomic structure (topics and subtopics)
- Deliberately inserted attractor patterns
- Control over the strength and distribution of attractors

### 2. Attractor Analysis

The `OllamaAttractorAnalyzer.py` script analyzes how attractors influence model behavior:
- Trains different classifiers on the synthetic data
- Compares performance on examples with and without attractors
- Evaluates the impact of taxonomic information on model performance
- Visualizes model performance across different scenarios

### 3. Pattern Recognition

The `AttractorPatternRecognizer.py` script identifies specific patterns that act as attractors:
- Analyzes n-grams, phrases, and topic-specific patterns
- Calculates attractor scores for different patterns
- Visualizes the distribution and impact of different attractor patterns

### 4. Visualization Tool

The `OllamaVisualizationTool.py` script provides tools to visualize:
- How attractors cluster in embedding space
- The "pull" that attractors exert on nearby examples
- Distances between examples with and without attractors
- How taxonomic relationships interact with attractor patterns

### 5. Experiment Runner

The `OllamaStrongAttractorExperimentRunner.py` script ties everything together:
- Runs experiments with different attractor strengths
- Compares models with and without taxonomic information
- Visualizes how increasing attractor strength impacts performance
- Demonstrates the value of taxonomic labels in mitigating attractor issues

### 6. Results Analyzer

The `results_analyzer.py` script helps analyze experimental results:
- Creates comprehensive summary reports
- Generates enhanced visualizations
- Provides an interactive explorer for examining examples
- Analyzes custom text for attractor patterns

## Getting Started

1. Set up Ollama by following the instructions in `OllamaSetupGuide.md`

2. Generate synthetic data:
```bash
python OllamaBasedSyntheticTrainingDataGenerator.py
```

3. Run the experiment:
```bash
python OllamaStrongAttractorExperimentRunner.py
```

For a quicker run without regenerating data:
```bash
python OllamaStrongAttractorExperimentRunner.py --skip-generation
```

4. Analyze the results:
```bash
python results_analyzer.py
```

## What to Look For

When running the experiments, pay attention to:

1. **Accuracy Gap**: The difference in model performance between examples with attractors vs. without attractors. A larger gap indicates the model is being strongly influenced by attractors.

2. **Feature Space Clustering**: Notice how examples with attractors tend to cluster more tightly in the feature space, creating "gravity wells" that can pull model decision boundaries.

3. **Taxonomy Effects**: Observe how adding taxonomic information helps models maintain more consistent performance across examples with and without attractors.

4. **Attractor Strength Impact**: See how increasing the strength of attractors amplifies their effect on model behavior, and how taxonomic information acts as a counterbalance.

## Requirements

- Python 3.x
- Ollama (locally installed)
- pandas
- numpy 
- matplotlib
- scikit-learn
- seaborn
- requests
- tqdm

## Conclusion

This project provides a practical demonstration of how to identify, visualize, and address the challenge of strong attractors in machine learning data. The visualization and analytical tools can be extended to real-world datasets to improve model robustness and fairness.

By understanding how strong attractors influence model behavior and leveraging taxonomic relationships to mitigate their effects, we can build more reliable and balanced machine learning systems.
