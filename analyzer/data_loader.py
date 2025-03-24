"""
Data loading utilities for the Strong Attractor Results Analyzer
"""

import os
import pandas as pd
import re
from collections import Counter

class DataLoader:
    """Class for loading and processing experiment data"""
    
    def __init__(self, data_path="synthetic_web_content_ollama.csv", 
                 output_dir="outputs", result_dir="analysis_results"):
        """
        Initialize the data loader
        
        Parameters:
        -----------
        data_path: str
            Path to the CSV file containing the data
        output_dir: str
            Directory containing experiment outputs
        result_dir: str
            Directory to save analysis results
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Initialize containers
        self.data = None
        self.llm_analysis = None
        self.metrics = {}
        self.patterns = {}
        
        # Load data
        self._load_data()
        self._load_llm_analysis()
        self._load_metrics()
        self._load_patterns()
    
    def _load_data(self):
        """Load the original data from CSV"""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Loaded {len(self.data)} examples from {self.data_path}")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def _load_llm_analysis(self):
        """Load LLM analysis if available"""
        llm_path = os.path.join(self.output_dir, "attractor_llm_analysis.csv")
        if os.path.exists(llm_path):
            try:
                self.llm_analysis = pd.read_csv(llm_path)
                print(f"Loaded LLM analysis with {len(self.llm_analysis)} examples")
            except Exception as e:
                print(f"Error loading LLM analysis: {e}")
    
    def _load_metrics(self):
        """Load metrics from experiment output"""
        for prefix in ["with_taxonomy_", "without_taxonomy_"]:
            metrics_path = os.path.join(self.output_dir, f"{prefix}metrics.csv")
            if os.path.exists(metrics_path):
                try:
                    self.metrics[prefix.replace("_", "")] = pd.read_csv(metrics_path).iloc[0].to_dict()
                    print(f"Loaded metrics for {prefix.replace('_', '')}")
                except Exception as e:
                    print(f"Error loading metrics for {prefix}: {e}")
    
    def _load_patterns(self):
        """Load attractor patterns from summary file"""
        patterns_path = os.path.join(self.output_dir, "attractor_patterns_summary.txt")
        if os.path.exists(patterns_path):
            try:
                with open(patterns_path, 'r') as f:
                    content = f.read()
                
                print(f"Patterns file loaded, content length: {len(content)}")
                
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
                        print(f"Found pattern section: '{current_section}'")
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
                                except Exception as e:
                                    print(f"Error parsing pattern line: {e}")
                
                # Count found patterns
                total_patterns = sum(len(patterns) for patterns in self.patterns.values())
                print(f"Total patterns found: {total_patterns}")
                
                if not self.patterns:
                    # As a fallback, create some sample patterns for testing
                    print("No patterns found, creating sample patterns")
                    self._create_sample_patterns()
            except Exception as e:
                print(f"Error loading patterns: {e}")
                # Create some sample patterns as fallback
                self._create_sample_patterns()
        else:
            print(f"Patterns file not found at {patterns_path}")
            self._create_sample_patterns()
    
    def _create_sample_patterns(self):
        """Create sample patterns as fallback"""
        self.patterns = {
            "1-grams": [("science", 0.8), ("technology", 0.7), ("business", 0.6)],
            "2-grams": [("artificial intelligence", 0.9), ("machine learning", 0.8)],
            "Phrase Patterns": [("the future of", 0.7), ("one of the most", 0.6)]
        }
        print("Created sample patterns for testing")
    
    def get_strongest_patterns(self, n=5):
        """
        Get the strongest patterns across all categories
        
        Parameters:
        -----------
        n: int
            Number of patterns to return
            
        Returns:
        --------
        List of (pattern, score, category) tuples
        """
        all_patterns = []
        for category, patterns in self.patterns.items():
            for pattern, score in patterns:
                all_patterns.append((pattern, score, category))
        
        # Sort by score in descending order
        all_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return all_patterns[:n]
