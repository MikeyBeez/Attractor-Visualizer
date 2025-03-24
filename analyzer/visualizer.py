"""
Visualization utilities for the Strong Attractor Results Analyzer
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import matplotlib.cm as cm
import shutil

# Set plotting style - use a style that works with newer matplotlib versions
try:
    plt.style.use('seaborn-whitegrid')  # For newer versions
except:
    try:
        plt.style.use('seaborn-v0_8-whitegrid')  # For older versions
    except:
        # If both fail, don't set a specific style
        pass

# Set color palette
sns.set_palette("viridis")

class Visualizer:
    """Class for creating visualizations of experiment results"""
    
    def __init__(self, data, llm_analysis, metrics, patterns, output_dir, result_dir):
        """
        Initialize the visualizer
        
        Parameters:
        -----------
        data: DataFrame
            The experiment data
        llm_analysis: DataFrame or None
            The LLM analysis data if available
        metrics: dict
            The metrics from the experiment
        patterns: dict
            The attractor patterns
        output_dir: str
            Directory containing experiment outputs
        result_dir: str
            Directory to save analysis results
        """
        self.data = data
        self.llm_analysis = llm_analysis
        self.metrics = metrics
        self.patterns = patterns
        self.output_dir = output_dir
        self.result_dir = result_dir
    
    def create_all_visualizations(self):
        """Create all available visualizations"""
        self.visualize_topic_attractor_distribution()
        self.visualize_feature_importance()
        self.visualize_pattern_distribution()
        self.create_attractor_dashboard()
        if self.llm_analysis is not None:
            self.visualize_llm_analysis()
        
        print("All visualizations created in", self.result_dir)
    
    def visualize_topic_attractor_distribution(self):
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
    
    def visualize_feature_importance(self):
        """Create a visualization of important features for attractor detection"""
        # Look for feature importance plots
        try:
            for prefix in ["with_taxonomy_", "without_taxonomy_"]:
                feature_path = os.path.join(self.output_dir, f"{prefix}attractor_features.png")
                if os.path.exists(feature_path):
                    # Copy to results dir with a more descriptive name
                    dest_path = os.path.join(self.result_dir, f"feature_importance_{prefix.replace('_', '')}.png")
                    shutil.copy(feature_path, dest_path)
                    print(f"Copied feature importance visualization to {dest_path}")
            
            print("Feature importance visualizations copied")
        except Exception as e:
            print(f"Error copying feature importance visualizations: {e}")
    
    def visualize_pattern_distribution(self):
        """Visualize the distribution of identified patterns"""
        if not self.patterns:
            print("No patterns to visualize")
            return
        
        try:
            # Create a visualization of pattern scores by category
            plt.figure(figsize=(14, 10))
            
            # Determine the number of subplots
            n_categories = len(self.patterns)
            if n_categories == 0:
                print("No pattern categories found")
                return
                
            n_cols = min(2, n_categories)
            n_rows = (n_categories + n_cols - 1) // n_cols
            
            subplot_count = 0
            for category, patterns in self.patterns.items():
                if not patterns:
                    continue
                
                subplot_count += 1
                if subplot_count > n_rows * n_cols:
                    print(f"Warning: Too many pattern categories, skipping {category}")
                    continue
                    
                ax = plt.subplot(n_rows, n_cols, subplot_count)
                
                # Get top patterns and scores
                top_patterns = patterns[:10]
                labels = [p[0][:20] + "..." if len(p[0]) > 20 else p[0] for p in top_patterns]
                scores = [p[1] for p in top_patterns]
                
                if not labels:
                    ax.text(0.5, 0.5, f"No patterns in {category}", 
                            horizontalalignment='center', verticalalignment='center')
                    ax.set_title(f"No Patterns: {category}")
                    continue
                
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
        except Exception as e:
            print(f"Error creating pattern distribution visualization: {e}")
    
    def visualize_llm_analysis(self):
        """Create visualizations from the LLM analysis"""
        if self.llm_analysis is None:
            return
        
        try:
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
            
            # Check if we have data to plot
            if attractor_counts:
                # Attractor words
                ax1.barh([w[0] for w in attractor_counts], [w[1] for w in attractor_counts])
                ax1.set_title("Common Words in Attractor Analysis")
                ax1.set_xlabel("Frequency")
            else:
                ax1.text(0.5, 0.5, "No attractor analysis data", 
                        horizontalalignment='center', verticalalignment='center')
                ax1.set_title("Attractor Analysis Words")
                
            if non_attractor_counts:
                # Non-attractor words
                ax2.barh([w[0] for w in non_attractor_counts], [w[1] for w in non_attractor_counts])
                ax2.set_title("Common Words in Non-Attractor Analysis")
                ax2.set_xlabel("Frequency")
            else:
                ax2.text(0.5, 0.5, "No non-attractor analysis data", 
                        horizontalalignment='center', verticalalignment='center')
                ax2.set_title("Non-Attractor Analysis Words")
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.result_dir, "llm_analysis_words.png"))
            plt.close()
            
            # Check if we have enough data for the heatmap
            if not self.llm_analysis.empty and 'category' not in self.llm_analysis.columns:
                # Combine topics and has_attractor into a single column
                self.llm_analysis['category'] = self.llm_analysis['topic'] + '_' + self.llm_analysis['has_attractor'].astype(str)
            
            if 'category' in self.llm_analysis.columns:
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
                
                # Only proceed if we have data
                if category_words and any(category_words.values()):
                    # Create a matrix of word frequencies
                    all_words = set()
                    for counter in category_words.values():
                        all_words.update(counter.keys())
                    
                    # Filter to top words across all categories
                    all_word_counts = Counter()
                    for counter in category_words.values():
                        all_word_counts.update(counter)
                    
                    top_words = [w for w, _ in all_word_counts.most_common(30)]
                    
                    if top_words and categories.size > 0:
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
                    else:
                        print("Not enough data for LLM analysis heatmap")
                else:
                    print("No category word data for LLM analysis heatmap")
            else:
                print("No category column in LLM analysis data")
            
            print("LLM analysis visualizations created")
        except Exception as e:
            print(f"Error creating LLM analysis visualizations: {e}")
    
    def create_attractor_dashboard(self):
        """Create a comprehensive dashboard of attractor information"""
        try:
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
        except Exception as e:
            print(f"Error creating attractor dashboard: {e}")
