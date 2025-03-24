#!/usr/bin/env python3
"""
Main entry point for the Strong Attractor Results Analyzer

This script analyzes and visualizes the results from the strong attractor experiment.
"""

import argparse

from analyzer.data_loader import DataLoader
from analyzer.visualizer import Visualizer
from analyzer.report_generator import ReportGenerator
from analyzer.text_analyzer import TextAnalyzer
from analyzer.explorer import Explorer

def main():
    """Main entry point for the analyzer"""
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
    
    # Initialize data loader
    data_loader = DataLoader(args.data, args.output, args.result_dir)
    
    # Initialize text analyzer
    text_analyzer = TextAnalyzer(data_loader.patterns)
    
    # Run selected mode(s)
    if args.analyze_text:
        text_analyzer.analyze_text(args.analyze_text, args.topic, None, args.ollama)
    elif args.summary:
        report_generator = ReportGenerator(
            data_loader.data, 
            data_loader.llm_analysis, 
            data_loader.metrics, 
            data_loader.patterns, 
            args.result_dir
        )
        report_generator.create_summary_report()
    elif args.visualize:
        visualizer = Visualizer(
            data_loader.data, 
            data_loader.llm_analysis, 
            data_loader.metrics, 
            data_loader.patterns, 
            args.output, 
            args.result_dir
        )
        visualizer.create_all_visualizations()
    elif args.interactive:
        explorer = Explorer(
            data_loader.data, 
            data_loader.llm_analysis, 
            data_loader.patterns, 
            text_analyzer
        )
        explorer.run()
    else:
        # Run all analyses
        print("\nGenerating summary report...")
        report_generator = ReportGenerator(
            data_loader.data, 
            data_loader.llm_analysis, 
            data_loader.metrics, 
            data_loader.patterns, 
            args.result_dir
        )
        report_generator.create_summary_report()
        
        print("\nCreating enhanced visualizations...")
        visualizer = Visualizer(
            data_loader.data, 
            data_loader.llm_analysis, 
            data_loader.metrics, 
            data_loader.patterns, 
            args.output, 
            args.result_dir
        )
        visualizer.create_all_visualizations()
        
        # Ask if user wants to run interactive explorer
        run_interactive = input("\nWould you like to run the interactive explorer? (y/n): ").lower() == 'y'
        if run_interactive:
            explorer = Explorer(
                data_loader.data, 
                data_loader.llm_analysis, 
                data_loader.patterns, 
                text_analyzer
            )
            explorer.run()

if __name__ == "__main__":
    main()
