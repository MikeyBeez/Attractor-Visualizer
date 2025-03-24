"""
Report generation utilities for the Strong Attractor Results Analyzer
"""

import os

class ReportGenerator:
    """Class for generating summary reports of experiment results"""
    
    def __init__(self, data, llm_analysis, metrics, patterns, result_dir):
        """
        Initialize the report generator
        
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
        result_dir: str
            Directory to save analysis results
        """
        self.data = data
        self.llm_analysis = llm_analysis
        self.metrics = metrics
        self.patterns = patterns
        self.result_dir = result_dir
    
    def create_summary_report(self):
        """
        Generate a summary report of all findings
        
        Returns:
        --------
        Path to the generated report
        """
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
        
        # Check if taxonomy made a difference
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
