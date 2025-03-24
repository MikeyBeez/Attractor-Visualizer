# Strong Attractor Results Analyzer

This tool helps you analyze and visualize the results from your strong attractor experiment. It provides a comprehensive view of how attractors influence your models and data.

## Features

- **Summary Report Generation**: Creates a detailed report of your experiment findings in Markdown format
- **Enhanced Visualizations**: Generates additional visualizations to better understand your data
- **Interactive Explorer**: Lets you browse and analyze specific examples and patterns
- **Custom Text Analysis**: Analyze any text to check for attractor patterns
- **LLM Integration**: Use your Ollama models for enhanced analysis

## Usage

### Basic Usage

To generate a summary report and visualizations:

```bash
python results_analyzer.py
```

This will:
1. Load your experiment data
2. Generate a comprehensive summary report
3. Create enhanced visualizations
4. Ask if you want to run the interactive explorer

### Command Line Options

```bash
python results_analyzer.py [OPTIONS]
```

Available options:

- `--data PATH`: Path to your data CSV (default: synthetic_web_content_ollama.csv)
- `--output DIR`: Path to the output directory where your experiment results are stored (default: outputs)
- `--result-dir DIR`: Directory to save analysis results (default: analysis_results)
- `--summary`: Generate summary report only
- `--visualize`: Create enhanced visualizations only
- `--interactive`: Run interactive explorer
- `--analyze-text TEXT`: Analyze a specific text for attractors
- `--topic TOPIC`: Topic for text analysis (optional)
- `--ollama MODEL`: Ollama model to use for analysis (optional)

### Examples

Generate only the summary report:
```bash
python results_analyzer.py --summary
```

Create visualizations only:
```bash
python results_analyzer.py --visualize
```

Run the interactive explorer:
```bash
python results_analyzer.py --interactive
```

Analyze a specific text:
```bash
python results_analyzer.py --analyze-text "Your text to analyze" --topic "technology" --ollama "deepseek-r1"
```

## Output

The analyzer creates a directory called `analysis_results` (or your custom name) with:

1. **summary_report.md**: A comprehensive report of your experiment findings
2. **topic_attractor_heatmap.png**: Heatmap showing distribution of attractors by topic
3. **topic_attractor_counts.png**: Bar chart of examples by topic and attractor status
4. **pattern_scores.png**: Visualization of attractor pattern scores by category
5. **llm_analysis_words.png**: Common words in LLM analysis
6. **attractor_dashboard.png**: Comprehensive dashboard of attractor information

## Interactive Explorer

The interactive explorer lets you:

1. View random examples from your dataset
2. View examples by ID
3. View examples by topic
4. View examples with the strongest attractors
5. Analyze custom text for attractors

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- requests (for Ollama integration)

## Tips

- For the best experience, run on a machine with Ollama installed
- The analyzer works with the output from the OllamaStrongAttractorExperimentRunner.py script
- Use the interactive explorer to dive deep into specific examples
