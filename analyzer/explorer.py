"""
Interactive explorer for the Strong Attractor Results Analyzer
"""

import random
import requests

class Explorer:
    """Interactive explorer for analyzing attractor examples"""
    
    def __init__(self, data, llm_analysis, patterns, text_analyzer):
        """
        Initialize the explorer
        
        Parameters:
        -----------
        data: DataFrame
            The experiment data
        llm_analysis: DataFrame or None
            The LLM analysis data if available
        patterns: dict
            The attractor patterns
        text_analyzer: TextAnalyzer
            The text analyzer for analyzing custom text
        """
        self.data = data
        self.llm_analysis = llm_analysis
        self.patterns = patterns
        self.text_analyzer = text_analyzer
    
    def run(self):
        """Run the interactive explorer"""
        try:
            # Check if running in a notebook
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
                    self.text_analyzer.analyze_text(text, topic, None, ollama_model)
                
                elif choice == '6':
                    # Exit
                    print("Exiting explorer")
                    break
                
                else:
                    print("Invalid choice. Please enter a number between 1 and 6.")
    
    def _display_example(self, example):
        """
        Display an example with its details
        
        Parameters:
        -----------
        example: Series
            The example to display
        """
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
            self.text_analyzer.analyze_text(
                example['content'], 
                example['topic'], 
                example['subtopic'],
                ollama_model
            )
