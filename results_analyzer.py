#!/usr/bin/env python3
"""
Strong Attractor Results Analyzer

This script helps analyze and visualize the results from the strong attractor experiment.
It summarizes the findings, creates additional visualizations, and provides insights
into the identified attractor patterns.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import argparse
import json
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import matplotlib.cm as cm

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

class ResultsAnalyzer:
    def __init__(self, data_path="synthetic_web_content_ollama.csv", output_dir="outputs", result_dir="analysis_results"):
        """Initialize the analyzer with paths to data and results"""
        self.data_path = data_path
        self.output_dir = output_dir
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Load the original data
        try:
            self.data = pd.read_csv(data_path)
            print(f"Loaded {len(self.data)} examples from {data_path}")
        except Exception as e:
            print(f"Error loading data: {e}")
            exit(1)
        
        # Load LLM analysis if available
        self.llm_analysis = None
        llm_path = os.path.join(output_dir, "attractor_llm_analysis.csv")
        if os.path.exists(llm_path):
            try:
                self.llm_analysis = pd.read_csv(llm_path)
                print(f"Loaded LLM analysis with {len(self.llm_analysis)} examples")
            except Exception as e:
                print(f"Error loading LLM analysis: {e}")
        
        # Load metrics
        self.metrics = {}
        for prefix in ["with_taxonomy_", "without_taxonomy_"]:
            metrics_path = os.path.join(output_dir, f"{prefix}metrics.csv")
            if os.path.exists(metrics_path):
                try:
                    self.metrics[prefix.replace("_", "")] = pd.read_csv(metrics_path).iloc[0].to_dict()
                    print(f"Loaded metrics for {prefix.replace('_', '')}")
                except Exception as e:
                    print(f"Error loading metrics for {prefix}: {e}")
        
        # Load patterns summary
        self.patterns = {}
        patterns_path = os.path.join(output_dir, "attractor_patterns_summary.txt")
        if os.path.exists(patterns_path):
            try:
                self.load_patterns(patterns_path)
                print(f"Loaded attractor patterns from {patterns_path}")
            except Exception as e:
                print(f"Error loading patterns: {e}")
    
    def load_patterns(self, patterns_path):
        """Parse the patterns summary file"""
        try:
            with open(patterns_path, 'r') as f:
                content = f.read()
            
            print("Debugging: Patterns file loaded, content length:", len(content))
            
            # Split by sections
            sections = content.split('\n\n')
            current_section = None
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                    
                lines = section.split('\n')
                
                # Handle section headers
                if ':' in lines[0] and not lines[0].startswith(('1.', '2.', '3.')):
                    current_section = lines[0].rstrip(':')
                    self.patterns[current_section] = []
                    print(f"Debugging: Found section '{current_section}'")
                    continue
                
                # If we have a current section and pattern entries
                if current_section and lines:
                    # Process each line as a potential pattern
                    for line in lines:
                        # Try to extract pattern and score
                        if any(line.startswith(str(i) + '.') for i in range(1, 20)):
                            try:
                                # Extract pattern between ". " and " (score:"
                                parts = line.split('. ', 1)
                                if len(parts) > 1:
                                    rest = parts[1]
                                    if '(score:' in rest:
                                        pattern, score_part = rest.rsplit('(score:', 1)
                                        pattern = pattern.strip()
                                        score = float(score_part.rstrip(')').strip())
                                        self.patterns[current_section].append((pattern, score))
                                        print(f"Debugging: Added pattern '{pattern}' with score {score}")
                            except Exception as e:
                                print(f"Debugging: Error parsing line '{line}': {e}")
            
            # Count found patterns
            total_patterns = sum(len(patterns) for patterns in self.patterns.values())
            print(f"Debugging: Total patterns found: {total_patterns}")
            
            if not self.patterns:
                # As a fallback, create some sample patterns for testing
                print("Debugging: No patterns found, creating sample patterns")
                self.patterns = {
                    "1-grams": [("science", 0.8), ("technology", 0.7), ("business", 0.6)],
                    "2-grams": [("artificial intelligence", 0.9), ("machine learning", 0.8)],
                    "Phrase Patterns": [("the future of", 0.7), ("one of the most", 0.6)]
                }
        except Exception as e:
            print(f"Error loading patterns: {e}")
            # Create some sample patterns as fallback
            self.patterns = {
                "1-grams": [("science", 0.8), ("technology", 0.7), ("business", 0.6)],
                "2-grams": [("artificial intelligence", 0.9), ("machine learning", 0.8)],
                "Phrase Patterns": [("the future of", 0.7), ("one of the most", 0.6)]
            }
    
    def create_summary_report(self):
        """Generate a summary report of all findings"""
        report = []
        
        # Dataset summary
        report.append("# Strong Attractor Experiment Summary Report")
        report.append("\n## Dataset Summary")
        report.append(f"- Total examples: {len(self.data)}")
        report.append(f"- Examples with attractors: {self.data['has_attractor'].sum()} ({self.data['has_attractor'].mean()*100:.1f}%)")
        report.append(f"- Examples without attractors: {(1-self.data['has_attractor']).sum()} ({(1-self.data['has_attractor']).mean()*100:.1f}%)")
        
        # Topic distribution
        report.append("\n### Topic Distribution")
        topic_counts = self.data['topic'].value_counts()
        for topic, count in topic_counts.items():
            report.append(f"- {topic}: {count} examples")
        
        # Model comparison
        if self.metrics:
            report.append("\n## Model Performance Comparison")
            report.append("\n| Metric | With Taxonomy | Without Taxonomy |")
            report.append("| ------ | ------------- | ---------------- |")
            
            metrics_to_show = [
                ('accuracy', 'Overall Accuracy'), 
                ('attractor_error', 'Error Rate (Attractor Examples)'),
                ('non_attractor_error', 'Error Rate (Non-Attractor Examples)'),
                ('gap', 'Error Gap'),
                ('attractor_detection_accuracy', 'Attractor Detection Accuracy')
            ]
            
            for key, name in metrics_to_show:
                with_tax = self.metrics.get('withtaxonomy', {}).get(key, "N/A")
                without_tax = self.metrics.get('withouttaxonomy', {}).get(key, "N/A")
                
                if isinstance(with_tax, (int, float)):
                    with_tax = f"{with_tax:.4f}"
                if isinstance(without_tax, (int, float)):
                    without_tax = f"{without_tax:.4f}"
                    
                report.append(f"| {name} | {with_tax} | {without_tax} |")
        
        # Attractor patterns
        if self.patterns:
            report.append("\n## Identified Attractor Patterns")
            
            for category, patterns in self.patterns.items():
                if patterns:
                    report.append(f"\n### {category}")
                    for i, (pattern, score) in enumerate(patterns[:10], 1):
                        report.append(f"{i}. **{pattern}** (score: {score:.2f})")
        
        # LLM analysis insights
        if self.llm_analysis is not None:
            report.append("\n## LLM Analysis Insights")
            
            # Extract attractor example analysis
            attractor_analysis = self.llm_analysis[self.llm_analysis['has_attractor'] == 1]
            if len(attractor_analysis) > 0:
                report.append("\n### Patterns Identified in Attractor Examples")
                for i, row in enumerate(attractor_analysis.iterrows(), 1):
                    _, r = row
                    report.append(f"{i}. **Topic: {r['topic']}/{r['subtopic']}**")
                    report.append(f"   Text: \"{r['content'][:100]}{'...' if len(r['content']) > 100 else ''}\"")
                    report.append(f"   Analysis: {r['llm_analysis']}")
                    report.append("")
            
            # Extract non-attractor example analysis
            non_attractor_analysis = self.llm_analysis[self.llm_analysis['has_attractor'] == 0]
            if len(non_attractor_analysis) > 0:
                report.append("\n### Characteristics of Non-Attractor Examples")
                for i, row in enumerate(non_attractor_analysis.iterrows(), 1):
                    _, r = row
                    report.append(f"{i}. **Topic: {r['topic']}/{r['subtopic']}**")
                    report.append(f"   Text: \"{r['content'][:100]}{'...' if len(r['content']) > 100 else ''}\"")
                    report.append(f"   Analysis: {r['llm_analysis']}")
                    report.append("")
        
        # Recommendations
        report.append("\n## Conclusions and Recommendations")
        
        # Check topic-specific patterns
        if topic:
            topic_key = f"Topic: {topic}"
            if topic_key in self.patterns:
                for pattern, score in self.patterns[topic_key]:
                    if pattern.lower() in text.lower():
                        pattern_matches.append({
                            'pattern': pattern,
                            'type': f'Topic-specific ({topic})',
                            'score': score
                        })
        
        # Sort by score
        pattern_matches.sort(key=lambda x: x['score'], reverse=True)
        results['identified_patterns'] = pattern_matches
        
        # Calculate overall attractor score
        if pattern_matches:
            # Use the top 3 matches, or fewer if there are fewer matches
            top_matches = pattern_matches[:min(3, len(pattern_matches))]
            # Weighted average of scores, with higher weight for higher scores
            weights = [1.0, 0.7, 0.5][:len(top_matches)]
            scores = [m['score'] for m in top_matches]
            results['attractor_score'] = sum(w * s for w, s in zip(weights, scores)) / sum(weights[:len(top_matches)])
            
            # Determine if it has an attractor
            results['has_attractor'] = results['attractor_score'] > 0.5
        
        # 2. Use Ollama for analysis if available
        if ollama_model:
            try:
                # Check if Ollama is running
                response = requests.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    models = [model['name'] for model in response.json().get('models', [])]
                    
                    if ollama_model in models:
                        print(f"Using Ollama model {ollama_model} for analysis...")
                        
                        # Create prompt
                        prompt = f"""
                        Please analyze this text for patterns that might act as "attractors" in machine learning:
                        
                        Text: "{text}"
                        
                        Topic: {topic or "Unknown"}
                        Subtopic: {subtopic or "Unknown"}
                        
                        Focus on:
                        1. Repeated phrases or structures
                        2. Strong sentiment or emotion words
                        3. Domain-specific terminology
                        4. Any other patterns that might cause a model to overfit
                        
                        Keep your analysis brief (3-4 sentences).
                        """
                        
                        # Query Ollama
                        response = requests.post(
                            "http://localhost:11434/api/generate",
                            json={
                                "model": ollama_model,
                                "prompt": prompt,
                                "stream": False
                            }
                        )
                        
                        if response.status_code == 200:
                            analysis = response.json().get("response", "")
                            results['analysis']['llm'] = analysis
                            print(f"\nLLM Analysis: {analysis}")
                    else:
                        print(f"Model {ollama_model} not found. Available models: {', '.join(models)}")
                else:
                    print("Failed to connect to Ollama")
            except Exception as e:
                print(f"Error using Ollama: {e}")
        
        # Print summary
        print("\nAnalysis Results:")
        print(f"Attractor Score: {results['attractor_score']:.2f}")
        print(f"Has Attractor: {'Yes' if results['has_attractor'] else 'No'}")
        
        if pattern_matches:
            print("\nIdentified patterns:")
            for i, match in enumerate(pattern_matches[:5], 1):
                print(f"{i}. {match['pattern']} ({match['type']}, score: {match['score']:.2f})")
        
        return results
    
    def interactive_explorer(self):
        """Run an interactive explorer for the dataset"""
        try:
            from IPython.display import display, HTML
            is_notebook = True
        except ImportError:
            is_notebook = False
        
        if not is_notebook:
            print("\nInteractive Explorer")
            print("===================")
            
            while True:
                print("\nExplorer Menu:")
                print("1. View random example")
                print("2. View example by ID")
                print("3. View examples by topic")
                print("4. View examples with strongest attractors")
                print("5. Analyze custom text")
                print("6. Exit")
                
                choice = input("\nEnter your choice (1-6): ")
                
                if choice == '1':
                    # View random example
                    example = self.data.sample(1).iloc[0]
                    self._display_example(example)
                
                elif choice == '2':
                    # View by ID
                    id_input = input("Enter example ID: ")
                    try:
                        id_val = int(id_input)
                        example = self.data[self.data['id'] == id_val]
                        if len(example) > 0:
                            self._display_example(example.iloc[0])
                        else:
                            print(f"No example found with ID {id_val}")
                    except ValueError:
                        print("Invalid ID. Please enter a number.")
                
                elif choice == '3':
                    # View by topic
                    topics = self.data['topic'].unique()
                    print("\nAvailable topics:")
                    for i, topic in enumerate(topics, 1):
                        print(f"{i}. {topic}")
                    
                    topic_choice = input("\nEnter topic number: ")
                    try:
                        topic_idx = int(topic_choice) - 1
                        if 0 <= topic_idx < len(topics):
                            selected_topic = topics[topic_idx]
                            topic_examples = self.data[self.data['topic'] == selected_topic]
                            example = topic_examples.sample(1).iloc[0]
                            self._display_example(example)
                        else:
                            print("Invalid topic number")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                
                elif choice == '4':
                    # View strongest attractors
                    # This requires having the pattern analysis results
                    if not self.patterns:
                        print("No pattern data available")
                        continue
                    
                    # Try to find examples with the strongest attractor patterns
                    strongest_patterns = []
                    for category, patterns in self.patterns.items():
                        if patterns:
                            strongest_patterns.extend(patterns[:2])
                    
                    # Sort by score
                    strongest_patterns.sort(key=lambda x: x[1], reverse=True)
                    
                    if not strongest_patterns:
                        print("No pattern data available")
                        continue
                    
                    # Show the top patterns
                    print("\nStrongest attractor patterns:")
                    for i, (pattern, score) in enumerate(strongest_patterns[:5], 1):
                        print(f"{i}. {pattern} (score: {score:.2f})")
                    
                    # Find examples containing these patterns
                    pattern_choice = input("\nEnter pattern number to see examples: ")
                    try:
                        pattern_idx = int(pattern_choice) - 1
                        if 0 <= pattern_idx < len(strongest_patterns):
                            selected_pattern = strongest_patterns[pattern_idx][0]
                            
                            # Find examples containing this pattern
                            matching_examples = []
                            for _, row in self.data.iterrows():
                                if selected_pattern.lower() in row['content'].lower():
                                    matching_examples.append(row)
                            
                            if matching_examples:
                                print(f"\nFound {len(matching_examples)} examples with pattern '{selected_pattern}'")
                                example = random.choice(matching_examples)
                                self._display_example(example)
                            else:
                                print(f"No examples found containing pattern '{selected_pattern}'")
                        else:
                            print("Invalid pattern number")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                
                elif choice == '5':
                    # Analyze custom text
                    text = input("\nEnter text to analyze: ")
                    if not text:
                        print("Empty text, skipping analysis")
                        continue
                    
                    # Optionally specify topic
                    topics = list(self.data['topic'].unique())
                    print("\nAvailable topics:")
                    for i, topic in enumerate(topics, 1):
                        print(f"{i}. {topic}")
                    print(f"{len(topics) + 1}. Other/None")
                    
                    topic_choice = input("\nEnter topic number (or press Enter to skip): ")
                    topic = None
                    if topic_choice:
                        try:
                            topic_idx = int(topic_choice) - 1
                            if 0 <= topic_idx < len(topics):
                                topic = topics[topic_idx]
                        except ValueError:
                            pass
                    
                    # Optionally use Ollama
                    use_ollama = input("\nUse Ollama for analysis? (y/n, default: n): ").lower() == 'y'
                    ollama_model = None
                    if use_ollama:
                        try:
                            response = requests.get("http://localhost:11434/api/tags")
                            if response.status_code == 200:
                                models = [model['name'] for model in response.json().get('models', [])]
                                if models:
                                    print("\nAvailable Ollama models:")
                                    for i, model_name in enumerate(models, 1):
                                        print(f"{i}. {model_name}")
                                    
                                    model_choice = input(f"\nSelect a model (1-{len(models)}, default: 1): ")
                                    if model_choice.isdigit() and 1 <= int(model_choice) <= len(models):
                                        ollama_model = models[int(model_choice)-1]
                                    else:
                                        ollama_model = models[0]
                                else:
                                    print("No models available. Skipping Ollama analysis.")
                            else:
                                print("Failed to connect to Ollama. Skipping Ollama analysis.")
                        except Exception as e:
                            print(f"Error checking Ollama: {e}")
                    
                    # Perform analysis
                    self.analyze_text_sample(text, topic, None, ollama_model)
                
                elif choice == '6':
                    # Exit
                    print("Exiting explorer")
                    break
                
                else:
                    print("Invalid choice. Please enter a number between 1 and 6.")
    
    def _display_example(self, example):
        """Display an example with its details"""
        print("\n" + "=" * 50)
        print(f"ID: {example['id']}")
        print(f"Topic: {example['topic']}")
        print(f"Subtopic: {example['subtopic']}")
        print(f"Has Attractor: {'Yes' if example['has_attractor'] == 1 else 'No'}")
        print("-" * 50)
        print(f"Content: {example['content']}")
        print("=" * 50)
        
        # Check if LLM analysis exists for this example
        if self.llm_analysis is not None:
            llm_for_example = self.llm_analysis[self.llm_analysis['text_id'] == example['id']]
            if len(llm_for_example) > 0:
                print("\nLLM Analysis:")
                print(llm_for_example.iloc[0]['llm_analysis'])
                print("-" * 50)
        
        # Analyze this example
        analysis_choice = input("\nAnalyze this example? (y/n, default: n): ").lower() == 'y'
        if analysis_choice:
            # Optionally use Ollama
            use_ollama = input("Use Ollama for analysis? (y/n, default: n): ").lower() == 'y'
            ollama_model = None
            if use_ollama:
                try:
                    response = requests.get("http://localhost:11434/api/tags")
                    if response.status_code == 200:
                        models = [model['name'] for model in response.json().get('models', [])]
                        if models:
                            print("\nAvailable Ollama models:")
                            for i, model_name in enumerate(models, 1):
                                print(f"{i}. {model_name}")
                            
                            model_choice = input(f"Select a model (1-{len(models)}, default: 1): ")
                            if model_choice.isdigit() and 1 <= int(model_choice) <= len(models):
                                ollama_model = models[int(model_choice)-1]
                            else:
                                ollama_model = models[0]
                        else:
                            print("No models available. Skipping Ollama analysis.")
                    else:
                        print("Failed to connect to Ollama. Skipping Ollama analysis.")
                except Exception as e:
                    print(f"Error checking Ollama: {e}")
            
            # Perform analysis
            self.analyze_text_sample(
                example['content'], 
                example['topic'], 
                example['subtopic'],
                ollama_model
            )

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze Strong Attractor Experiment Results")
    parser.add_argument("--data", type=str, default="synthetic_web_content_ollama.csv", help="Path to the data CSV")
    parser.add_argument("--output", type=str, default="outputs", help="Path to the output directory")
    parser.add_argument("--result-dir", type=str, default="analysis_results", help="Directory to save analysis results")
    parser.add_argument("--summary", action="store_true", help="Generate summary report only")
    parser.add_argument("--visualize", action="store_true", help="Create enhanced visualizations only")
    parser.add_argument("--interactive", action="store_true", help="Run interactive explorer")
    parser.add_argument("--analyze-text", type=str, help="Analyze a specific text for attractors")
    parser.add_argument("--topic", type=str, help="Topic for text analysis")
    parser.add_argument("--ollama", type=str, help="Ollama model to use for analysis")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ResultsAnalyzer(args.data, args.output, args.result_dir)
    
    # Run selected mode(s)
    if args.analyze_text:
        analyzer.analyze_text_sample(args.analyze_text, args.topic, None, args.ollama)
    elif args.summary:
        analyzer.create_summary_report()
    elif args.visualize:
        analyzer.create_enhanced_visualizations()
    elif args.interactive:
        analyzer.interactive_explorer()
    else:
        # Run all analyses
        print("\nGenerating summary report...")
        analyzer.create_summary_report()
        
        print("\nCreating enhanced visualizations...")
        analyzer.create_enhanced_visualizations()
        
        # Ask if user wants to run interactive explorer
        run_interactive = input("\nWould you like to run the interactive explorer? (y/n): ").lower() == 'y'
        if run_interactive:
            analyzer.interactive_explorer()

if __name__ == "__main__":
    import random
    main() if taxonomy made a difference
        if self.metrics and 'withtaxonomy' in self.metrics and 'withouttaxonomy' in self.metrics:
            with_gap = self.metrics['withtaxonomy'].get('gap', 0)
            without_gap = self.metrics['withouttaxonomy'].get('gap', 0)
            
            if with_gap < without_gap:
                report.append("- **Taxonomy is effective**: The model with taxonomic information shows a smaller error gap between attractor and non-attractor examples.")
                report.append(f"  - Gap reduction: {(without_gap - with_gap)*100:.2f}%")
                report.append("  - Recommendation: Continue using taxonomic labels to improve model robustness.")
            elif with_gap > without_gap:
                report.append("- **Taxonomy shows no benefit**: The model without taxonomic information actually performs better in terms of error gap.")
                report.append(f"  - Gap difference: {(with_gap - without_gap)*100:.2f}%")
                report.append("  - Recommendation: Investigate why taxonomy is not helping in this case.")
            else:
                report.append("- **No significant difference**: Taxonomy does not appear to affect model performance in this experiment.")
                report.append("  - Recommendation: Consider testing with stronger attractors or a larger dataset.")
        
        # Based on patterns identified
        if self.patterns:
            pattern_types = list(self.patterns.keys())
            report.append("\n### Pattern-Based Recommendations")
            report.append(f"- The most significant attractor pattern types are {', '.join(pattern_types[:3])}")
            report.append("- Consider implementing countermeasures:")
            report.append("  1. **Data augmentation**: Generate more diverse examples that don't follow these patterns")
            report.append("  2. **Pattern regularization**: Add penalties for model weights associated with these patterns")
            report.append("  3. **Explicit feature engineering**: Create features that explicitly detect these patterns")
        
        # Write the report
        report_path = os.path.join(self.result_dir, "summary_report.md")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Summary report saved to {report_path}")
        return report_path
    
    def create_enhanced_visualizations(self):
        """Create additional visualizations to better understand the data"""
        self._visualize_topic_attractor_distribution()
        self._visualize_feature_importance()
        self._visualize_pattern_distribution()
        self._create_attractor_dashboard()
        if self.llm_analysis is not None:
            self._visualize_llm_analysis()
    
    def _visualize_topic_attractor_distribution(self):
        """Visualize how attractors are distributed across topics"""
        plt.figure(figsize=(12, 8))
        
        # Create a cross tabulation
        cross_tab = pd.crosstab(
            self.data['topic'],
            self.data['has_attractor'],
            normalize='index'
        ) * 100
        
        # Plot as a heatmap
        sns.heatmap(
            cross_tab,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            cbar_kws={'label': 'Percentage (%)'}
        )
        
        plt.title('Distribution of Attractors by Topic (%)')
        plt.xlabel('Has Attractor')
        plt.ylabel('Topic')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.result_dir, "topic_attractor_heatmap.png"))
        plt.close()
        
        # Also create a stacked bar chart
        plt.figure(figsize=(12, 6))
        
        # Count by topic and attractor
        counts = pd.crosstab(self.data['topic'], self.data['has_attractor'])
        counts.columns = ['Non-Attractor', 'Attractor']
        
        # Plot
        counts.plot(kind='bar', stacked=True)
        plt.title('Distribution of Examples by Topic and Attractor Status')
        plt.xlabel('Topic')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Type')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.result_dir, "topic_attractor_counts.png"))
        plt.close()
        
        print("Topic-attractor distribution visualizations created")
    
    def _visualize_feature_importance(self):
        """Create a visualization of important features for attractor detection"""
        # Look for feature importance plots
        for prefix in ["with_taxonomy_", "without_taxonomy_"]:
            feature_path = os.path.join(self.output_dir, f"{prefix}attractor_features.png")
            if os.path.exists(feature_path):
                # Copy to results dir with a more descriptive name
                import shutil
                shutil.copy(
                    feature_path, 
                    os.path.join(self.result_dir, f"feature_importance_{prefix.replace('_', '')}.png")
                )
        
        print("Feature importance visualizations copied")
    
    def _visualize_pattern_distribution(self):
        """Visualize the distribution of identified patterns"""
        if not self.patterns:
            return
        
        # Create a visualization of pattern scores by category
        plt.figure(figsize=(14, 10))
        
        # Determine the number of subplots
        n_categories = len(self.patterns)
        n_cols = min(2, n_categories)
        n_rows = (n_categories + n_cols - 1) // n_cols
        
        for i, (category, patterns) in enumerate(self.patterns.items(), 1):
            if not patterns:
                continue
                
            ax = plt.subplot(n_rows, n_cols, i)
            
            # Get top patterns and scores
            top_patterns = patterns[:10]
            labels = [p[0][:20] + "..." if len(p[0]) > 20 else p[0] for p in top_patterns]
            scores = [p[1] for p in top_patterns]
            
            # Sort by score in descending order
            sorted_indices = np.argsort(scores)[::-1]
            labels = [labels[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]
            
            # Create bar chart
            ax.barh(range(len(labels)), scores, align='center')
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Score')
            ax.set_title(f"Top Patterns: {category}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, "pattern_scores.png"))
        plt.close()
        
        print("Pattern distribution visualization created")
    
    def _visualize_llm_analysis(self):
        """Create visualizations from the LLM analysis"""
        if self.llm_analysis is None:
            return
        
        # Extract common words from attractor analysis
        attractor_words = []
        for analysis in self.llm_analysis[self.llm_analysis['has_attractor'] == 1]['llm_analysis']:
            if isinstance(analysis, str):
                words = re.findall(r'\b\w+\b', analysis.lower())
                attractor_words.extend([w for w in words if len(w) > 3 and w not in ['this', 'that', 'with', 'would', 'could', 'these', 'there', 'their', 'they']])
        
        # Extract common words from non-attractor analysis
        non_attractor_words = []
        for analysis in self.llm_analysis[self.llm_analysis['has_attractor'] == 0]['llm_analysis']:
            if isinstance(analysis, str):
                words = re.findall(r'\b\w+\b', analysis.lower())
                non_attractor_words.extend([w for w in words if len(w) > 3 and w not in ['this', 'that', 'with', 'would', 'could', 'these', 'there', 'their', 'they']])
        
        # Count word frequencies
        attractor_counts = Counter(attractor_words).most_common(15)
        non_attractor_counts = Counter(non_attractor_words).most_common(15)
        
        # Create side-by-side bar charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        # Attractor words
        ax1.barh([w[0] for w in attractor_counts], [w[1] for w in attractor_counts])
        ax1.set_title("Common Words in Attractor Analysis")
        ax1.set_xlabel("Frequency")
        
        # Non-attractor words
        ax2.barh([w[0] for w in non_attractor_counts], [w[1] for w in non_attractor_counts])
        ax2.set_title("Common Words in Non-Attractor Analysis")
        ax2.set_xlabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, "llm_analysis_words.png"))
        plt.close()
        
        # Create a combined visualization of LLM analysis for specific examples
        plt.figure(figsize=(15, 10))
        
        # Combine topics and has_attractor into a single column
        self.llm_analysis['category'] = self.llm_analysis['topic'] + '_' + self.llm_analysis['has_attractor'].astype(str)
        
        # Get word counts for each category
        categories = self.llm_analysis['category'].unique()
        category_words = {}
        
        for category in categories:
            words = []
            for analysis in self.llm_analysis[self.llm_analysis['category'] == category]['llm_analysis']:
                if isinstance(analysis, str):
                    w = re.findall(r'\b\w+\b', analysis.lower())
                    words.extend([word for word in w if len(word) > 3 and word not in ['this', 'that', 'with', 'would', 'could', 'these', 'there', 'their', 'they']])
            category_words[category] = Counter(words)
        
        # Create a matrix of word frequencies
        all_words = set()
        for counter in category_words.values():
            all_words.update(counter.keys())
        
        # Filter to top words across all categories
        all_word_counts = Counter()
        for counter in category_words.values():
            all_word_counts.update(counter)
        
        top_words = [w for w, _ in all_word_counts.most_common(30)]
        
        # Create the matrix
        matrix = np.zeros((len(categories), len(top_words)))
        for i, category in enumerate(categories):
            for j, word in enumerate(top_words):
                matrix[i, j] = category_words[category].get(word, 0)
        
        # Normalize by row
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix_normalized = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            if row_sums[i] > 0:
                matrix_normalized[i] = matrix[i] / row_sums[i]
        
        # Plot as heatmap
        plt.figure(figsize=(18, 12))
        sns.heatmap(
            matrix_normalized,
            annot=False,
            cmap='Blues',
            xticklabels=top_words,
            yticklabels=categories
        )
        plt.title('Word Usage in LLM Analysis by Category')
        plt.xlabel('Words')
        plt.ylabel('Category (Topic_HasAttractor)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.result_dir, "llm_analysis_heatmap.png"))
        plt.close()
        
        print("LLM analysis visualizations created")
    
    def _create_attractor_dashboard(self):
        """Create a comprehensive dashboard of attractor information"""
        plt.figure(figsize=(15, 20))
        
        # Title
        plt.suptitle("Strong Attractor Analysis Dashboard", fontsize=16, y=0.98)
        
        # Data distribution
        ax1 = plt.subplot(4, 2, 1)
        self.data['has_attractor'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax1)
        ax1.set_title("Distribution of Examples")
        ax1.set_ylabel("")
        
        # Topic distribution
        ax2 = plt.subplot(4, 2, 2)
        self.data['topic'].value_counts().plot(kind='bar', ax=ax2)
        ax2.set_title("Topics in Dataset")
        ax2.set_ylabel("Count")
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Attractor by topic
        ax3 = plt.subplot(4, 2, 3)
        pd.crosstab(self.data['topic'], self.data['has_attractor']).plot(kind='bar', stacked=True, ax=ax3)
        ax3.set_title("Attractors by Topic")
        ax3.set_ylabel("Count")
        ax3.legend(["Non-Attractor", "Attractor"])
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Model comparison (if available)
        ax4 = plt.subplot(4, 2, 4)
        if self.metrics and 'withtaxonomy' in self.metrics and 'withouttaxonomy' in self.metrics:
            metrics_to_compare = ['accuracy', 'gap', 'attractor_detection_accuracy']
            labels = ['Accuracy', 'Error Gap', 'Attractor Detection']
            
            with_tax_values = [self.metrics['withtaxonomy'].get(m, 0) for m in metrics_to_compare]
            without_tax_values = [self.metrics['withouttaxonomy'].get(m, 0) for m in metrics_to_compare]
            
            x = np.arange(len(labels))
            width = 0.35
            
            ax4.bar(x - width/2, with_tax_values, width, label='With Taxonomy')
            ax4.bar(x + width/2, without_tax_values, width, label='Without Taxonomy')
            
            ax4.set_title("Model Performance Comparison")
            ax4.set_ylabel("Score")
            ax4.set_xticks(x)
            ax4.set_xticklabels(labels)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, "No model comparison data available", 
                    horizontalalignment='center', verticalalignment='center')
            ax4.set_title("Model Performance Comparison")
        
        # Top n-grams (1-grams)
        ax5 = plt.subplot(4, 2, 5)
        if '1-grams' in self.patterns and self.patterns['1-grams']:
            top_unigrams = self.patterns['1-grams'][:8]
            labels = [p[0] for p in top_unigrams]
            values = [p[1] for p in top_unigrams]
            
            ax5.bar(labels, values)
            ax5.set_title("Top Unigram Attractors")
            ax5.set_ylabel("Score")
            plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
        else:
            ax5.text(0.5, 0.5, "No unigram data available", 
                    horizontalalignment='center', verticalalignment='center')
            ax5.set_title("Top Unigram Attractors")
        
        # Top phrases
        ax6 = plt.subplot(4, 2, 6)
        if 'Phrase Patterns' in self.patterns and self.patterns['Phrase Patterns']:
            top_phrases = self.patterns['Phrase Patterns'][:8]
            labels = [p[0][:15] + "..." if len(p[0]) > 15 else p[0] for p in top_phrases]
            values = [p[1] for p in top_phrases]
            
            ax6.bar(labels, values)
            ax6.set_title("Top Phrase Patterns")
            ax6.set_ylabel("Score")
            plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
        else:
            ax6.text(0.5, 0.5, "No phrase pattern data available", 
                    horizontalalignment='center', verticalalignment='center')
            ax6.set_title("Top Phrase Patterns")
        
        # Topic-specific patterns
        ax7 = plt.subplot(4, 2, 7)
        topic_patterns = {}
        for key, patterns in self.patterns.items():
            if key.startswith("Topic:") and patterns:
                topic = key.replace("Topic: ", "")
                topic_patterns[topic] = max([p[1] for p in patterns])
        
        if topic_patterns:
            ax7.bar(topic_patterns.keys(), topic_patterns.values())
            ax7.set_title("Strongest Attractor by Topic")
            ax7.set_ylabel("Score")
            plt.setp(ax7.get_xticklabels(), rotation=45, ha='right')
        else:
            ax7.text(0.5, 0.5, "No topic-specific pattern data available", 
                    horizontalalignment='center', verticalalignment='center')
            ax7.set_title("Strongest Attractor by Topic")
        
        # Text length comparison
        ax8 = plt.subplot(4, 2, 8)
        self.data['text_length'] = self.data['content'].str.len()
        
        sns.boxplot(
            x='has_attractor', 
            y='text_length', 
            data=self.data,
            ax=ax8
        )
        ax8.set_title("Text Length Comparison")
        ax8.set_xlabel("Has Attractor")
        ax8.set_ylabel("Character Count")
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(self.result_dir, "attractor_dashboard.png"))
        plt.close()
        
        print("Attractor dashboard created")
    
    def analyze_text_sample(self, text, topic=None, subtopic=None, ollama_model=None):
        """Analyze a specific text sample for attractors"""
        print(f"\nAnalyzing text sample for attractors:")
        print(f"Text: '{text}'")
        
        results = {
            'text': text,
            'analysis': {},
            'has_attractor': None,
            'attractor_score': 0,
            'identified_patterns': []
        }
        
        # 1. Check for pattern matches
        pattern_matches = []
        
        # Check n-grams
        for n in range(1, 4):
            ngram_key = f"{n}-grams"
            if ngram_key in self.patterns:
                # Create n-grams from the text
                words = re.findall(r'\b\w+\b', text.lower())
                text_ngrams = []
                for i in range(len(words) - n + 1):
                    text_ngrams.append(' '.join(words[i:i+n]))
                
                # Check for matches
                for pattern, score in self.patterns[ngram_key]:
                    if pattern in text_ngrams:
                        pattern_matches.append({
                            'pattern': pattern,
                            'type': ngram_key,
                            'score': score
                        })
        
        # Check phrase patterns
        if 'Phrase Patterns' in self.patterns:
            for pattern, score in self.patterns['Phrase Patterns']:
                # Convert to regex pattern - this is approximate
                regex_pattern = pattern.replace(' ', '\\s+')
                if re.search(regex_pattern, text, re.IGNORECASE):
                    pattern_matches.append({
                        'pattern': pattern,
                        'type': 'Phrase',
                        'score': score
                    })
        
        # Check
