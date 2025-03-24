# Attractor-Visualizer
Synthetic training data creatinon and visualization using taxonomic labels with the objective function of improved handling of strong attractor tokens

# Strong Attractor Model Experiment

This project explores how machine learning models handle "strong attractors" in training data and demonstrates the value of taxonomic labels in improving model robustness.

## Overview

Strong attractors are patterns in training data that can disproportionately influence model behavior, potentially causing overfitting or bias. This project provides tools to:

1. Generate synthetic training data with controlled attractor patterns
2. Visualize how attractors influence the distribution of examples in feature space
3. Evaluate how models perform on examples with and without attractors
4. Demonstrate how taxonomic relationships can help models handle attractors more effectively

## Components

### 1. Synthetic Data Generator

The `synthetic-data-generator.py` script creates a dataset mimicking web content with:
- Hierarchical taxonomic structure (topics and subtopics)
- Deliberately inserted attractor patterns
- Control over the strength and distribution of attractors

### 2. Attractor Pattern Analyzer

The `attractor-analyzer.py` script analyzes how attractors influence model behavior:
- Trains different classifiers on the synthetic data
- Compares performance on examples with and without attractors
- Evaluates the impact of taxonomic information on model performance
- Visualizes model performance across different scenarios

### 3. Interactive Visualization Tool

The `visualization-tool.py` script provides tools to visualize:
- How attractors cluster in embedding space
- The "pull" that attractors exert on nearby examples
- Distances between examples with and without attractors
- How taxonomic relationships interact with attractor patterns

### 4. Strong Attractor Experiment Runner

The `strong-attractor-experiment.py` script ties everything together:
- Runs experiments with different attractor strengths
- Compares models with and without taxonomic information
- Visualizes how increasing attractor strength impacts performance
- Demonstrates the value of taxonomic labels in mitigating attractor issues

## Running the Experiment

1. First, generate the synthetic data:
```python
python synthetic-data-generator.py
```

2. Visualize the data distribution and attractors:
```python
python visualization-tool.py
```

3. Analyze how models handle attractors:
```python
python attractor-analyzer.py
```

4. Run the full experiment:
```python
python strong-attractor-experiment.py
```

## What to Look For

When running the experiments, pay attention to:

1. **Accuracy Gap**: The difference in model performance between examples with attractors vs. without attractors. A larger gap indicates the model is being strongly influenced by attractors.

2. **Feature Space Clustering**: Notice how examples with attractors tend to cluster more tightly in the feature space, creating "gravity wells" that can pull model decision boundaries.

3. **Taxonomy Effects**: Observe how adding taxonomic information helps models maintain more consistent performance across examples with and without attractors.

4. **Attractor Strength Impact**: See how increasing the strength of attractors amplifies their effect on model behavior, and how taxonomic information acts as a counterbalance.

## Conclusion

This project provides a practical demonstration of how to identify, visualize, and address the challenge of strong attractors in machine learning data. The visualization and analytical tools can be extended to real-world datasets to improve model robustness and fairness.

By understanding how strong attractors influence model behavior and leveraging taxonomic relationships to mitigate their effects, we can build more reliable and balanced machine learning systems.
