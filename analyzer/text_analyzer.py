"""
Text analysis utilities for the Strong Attractor Results Analyzer
"""

import re
import requests

class TextAnalyzer:
    """Class for analyzing text samples for attractor patterns"""
    
    def __init__(self, patterns):
        """
        Initialize the text analyzer
        
        Parameters:
        -----------
        patterns: dict
            The attractor patterns
        """
        self.patterns = patterns
    
    def analyze_text(self, text, topic=None, subtopic=None, ollama_model=None):
        """
        Analyze a specific text sample for attractors
        
        Parameters:
        -----------
        text: str
            The text to analyze
        topic: str or None
            The topic of the text if known
        subtopic: str or None
            The subtopic of the text if known
        ollama_model: str or None
            Ollama model to use for analysis
            
        Returns:
        --------
        Dictionary with analysis results
        """
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
