# Setting Up Ollama for the Strong Attractor Project

This guide will help you set up Ollama on your Mac Mini to run local language models for our strong attractor experiment.

## What is Ollama?

Ollama is an open-source tool that lets you run large language models (LLMs) locally on your computer. It's designed to be lightweight and easy to use, making it perfect for running models on hardware with limited resources like your Mac Mini with 16GB of RAM.

## Installation Steps

### 1. Install Ollama

1. Download Ollama for macOS from the official website: [https://ollama.ai/download](https://ollama.ai/download)
2. Install the application by dragging it to your Applications folder
3. Launch Ollama from your Applications folder

Alternatively, you can install it using Homebrew:

```bash
brew install ollama
```

### 2. Start the Ollama Service

Once installed, start the Ollama service:

```bash
ollama serve
```

This will run Ollama in the background, listening on port 11434.

### 3. Pull a Model

For our project, we'll need a language model. Here are some good options that work well on a Mac Mini with 16GB of RAM:

```bash
# For best performance but more RAM usage (7-8GB)
ollama pull llama3

# For lower RAM usage (3-4GB)
ollama pull mistral

# For minimal RAM usage (2GB)
ollama pull gemma:2b

# DeepSeek model - good quality, medium RAM usage
ollama pull deepseek-r1
```

The smaller models will run faster on your Mac Mini but may produce less sophisticated text. For our experimental purposes, even the smaller models should work well.

### 4. Verify the Installation

Check that Ollama is running correctly:

```bash
ollama list
```

This should show you the models you've pulled.

## Recommended Models for Mac Mini (16GB RAM)

Here are some models that should work well on your Mac Mini:

1. **deepseek-r1** - High quality and efficient (~4-5GB RAM)
2. **llama3** - Good balance of performance and quality (~7-8GB RAM)
3. **mistral** - Efficient model with good results (~4GB RAM)
4. **gemma:2b** - Very lightweight model (~2GB RAM)
5. **phi** - Microsoft's small but capable model (~4GB RAM)

## Running Our Project with Ollama

Our project has been adapted to use Ollama for generating synthetic data with attractor patterns. The system will:

1. Use your locally running Ollama instance to generate text
2. Create examples with and without strong attractor patterns
3. Maintain the taxonomic structure we want to study
4. Allow us to experiment with different models to see how they generate attractors

## Python Environment Setup

For this project, you'll need a Python virtual environment with the necessary packages:

```bash
# Create a virtual environment
python -m venv attractor-env

# Activate the virtual environment
source attractor-env/bin/activate

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn requests tqdm
```

## Troubleshooting

If you encounter issues:

1. **"Failed to connect to Ollama"** - Make sure the Ollama service is running with `ollama serve`
2. **High memory usage** - Try using a smaller model like `gemma:2b` or `phi`
3. **Slow generation** - Be patient, local models are slower than cloud services but don't have usage limits
4. **Model not found** - Make sure you've pulled the model with `ollama pull <model_name>`
5. **Python package errors** - Make sure you're using the virtual environment by running `source attractor-env/bin/activate`

## Running the Entire Experiment

Once Ollama is set up and you have a model pulled, you can run the full experiment:

```bash
# Activate the virtual environment
source attractor-env/bin/activate

# Run the experiment
python OllamaStrongAttractorExperimentRunner.py
```

Data generation can take time (possibly an hour or more) depending on the model and number of examples. To skip data generation and use existing data:

```bash
python OllamaStrongAttractorExperimentRunner.py --skip-generation
```

To analyze the results:

```bash
python results_analyzer.py
```
