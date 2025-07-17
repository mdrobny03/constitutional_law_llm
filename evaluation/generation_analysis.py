"""
Evaluation and generation analysis for Constitutional Law LLM.
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
import logging

try:
    # Try relative imports first
    from ..src.model_utils import ModelManager
    from ..src.config import config
except ImportError:
    # Fall back to absolute imports
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from model_utils import ModelManager
    from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalResponseEvaluator:
    """Evaluates legal reasoning responses using various metrics."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def evaluate_response_quality(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Evaluate response quality using various metrics.
        
        Args:
            generated: Generated response
            reference: Reference response
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Basic length metrics
        metrics['generated_length'] = len(generated.split())
        metrics['reference_length'] = len(reference.split())
        metrics['length_ratio'] = metrics['generated_length'] / max(metrics['reference_length'], 1)
        
        # Simple overlap metrics
        generated_words = set(generated.lower().split())
        reference_words = set(reference.lower().split())
        
        intersection = generated_words & reference_words
        union = generated_words | reference_words
        
        metrics['word_overlap'] = len(intersection) / max(len(union), 1)
        metrics['precision'] = len(intersection) / max(len(generated_words), 1)
        metrics['recall'] = len(intersection) / max(len(reference_words), 1)
        
        # F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0
        
        # Legal reasoning indicators
        legal_terms = {
            'court', 'held', 'ruled', 'decided', 'constitutional', 'amendment',
            'precedent', 'statute', 'plaintiff', 'defendant', 'violated', 'protected',
            'rights', 'freedom', 'justice', 'opinion', 'dissent', 'majority'
        }
        
        generated_legal = generated_words & legal_terms
        reference_legal = reference_words & legal_terms
        
        metrics['legal_term_coverage'] = len(generated_legal) / max(len(reference_legal), 1)
        metrics['legal_term_precision'] = len(generated_legal) / max(len(generated_words), 1)
        
        return metrics
    
    def evaluate_constitutional_reasoning(self, response: str, case_type: str = "general") -> Dict[str, float]:
        """
        Evaluate constitutional reasoning quality.
        
        Args:
            response: Generated response
            case_type: Type of constitutional case (first_amendment, fourth_amendment, etc.)
            
        Returns:
            Dictionary of reasoning evaluation metrics
        """
        metrics = {}
        response_lower = response.lower()
        
        # Constitutional framework indicators
        constitutional_indicators = [
            'constitution', 'constitutional', 'amendment', 'bill of rights',
            'supreme court', 'precedent', 'judicial review'
        ]
        
        framework_score = sum(1 for indicator in constitutional_indicators if indicator in response_lower)
        metrics['constitutional_framework'] = framework_score / len(constitutional_indicators)
        
        # Legal reasoning structure
        reasoning_indicators = [
            'held that', 'ruled that', 'decided that', 'concluded that',
            'because', 'therefore', 'however', 'furthermore', 'in contrast'
        ]
        
        structure_score = sum(1 for indicator in reasoning_indicators if indicator in response_lower)
        metrics['reasoning_structure'] = min(structure_score / 3, 1.0)  # Normalize to max 1.0
        
        # Case-specific evaluation
        if case_type == "first_amendment":
            first_amendment_terms = [
                'free speech', 'freedom of speech', 'expression', 'religion',
                'establishment', 'free exercise', 'press', 'assembly', 'petition'
            ]
            amendment_score = sum(1 for term in first_amendment_terms if term in response_lower)
            metrics['amendment_specific'] = amendment_score / len(first_amendment_terms)
            
        elif case_type == "fourth_amendment":
            fourth_amendment_terms = [
                'search', 'seizure', 'warrant', 'probable cause', 'reasonable',
                'privacy', 'unreasonable search', 'exclusionary rule'
            ]
            amendment_score = sum(1 for term in fourth_amendment_terms if term in response_lower)
            metrics['amendment_specific'] = amendment_score / len(fourth_amendment_terms)
        
        else:
            metrics['amendment_specific'] = 0.0
        
        # Overall coherence (simple heuristic)
        sentences = response.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        metrics['coherence'] = min(avg_sentence_length / 20, 1.0)  # Normalize, prefer 10-20 words per sentence
        
        return metrics

class GenerationAnalyzer:
    """Analyzes generation quality across different parameters."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.evaluator = LegalResponseEvaluator(model_manager)
    
    def analyze_generation_parameters(self, test_cases: List[Dict[str, Any]], 
                                    param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Analyze generation quality across different parameters.
        
        Args:
            test_cases: List of test cases to evaluate
            param_grid: Grid of generation parameters to test
            
        Returns:
            Analysis results
        """
        results = []
        
        # Generate all parameter combinations
        import itertools
        param_combinations = [
            dict(zip(param_grid.keys(), combination))
            for combination in itertools.product(*param_grid.values())
        ]
        
        logger.info(f"Analyzing {len(param_combinations)} parameter combinations on {len(test_cases)} test cases")
        
        for params in tqdm(param_combinations, desc="Parameter combinations"):
            param_results = {
                'params': params,
                'case_results': [],
                'aggregate_metrics': {}
            }
            
            for case in test_cases:
                # Generate response with current parameters
                response = self.model_manager.generate_response(
                    case['facts'], 
                    case['question'], 
                    generation_config=params
                )
                
                # Evaluate response
                quality_metrics = self.evaluator.evaluate_response_quality(
                    response, 
                    case.get('reference', '')
                )
                
                reasoning_metrics = self.evaluator.evaluate_constitutional_reasoning(
                    response, 
                    case.get('type', 'general')
                )
                
                case_result = {
                    'case_id': case.get('id', ''),
                    'generated_response': response,
                    'quality_metrics': quality_metrics,
                    'reasoning_metrics': reasoning_metrics
                }
                
                param_results['case_results'].append(case_result)
            
            # Calculate aggregate metrics
            if param_results['case_results']:
                quality_keys = param_results['case_results'][0]['quality_metrics'].keys()
                reasoning_keys = param_results['case_results'][0]['reasoning_metrics'].keys()
                
                for key in quality_keys:
                    values = [case['quality_metrics'][key] for case in param_results['case_results']]
                    param_results['aggregate_metrics'][f'avg_{key}'] = np.mean(values)
                
                for key in reasoning_keys:
                    values = [case['reasoning_metrics'][key] for case in param_results['case_results']]
                    param_results['aggregate_metrics'][f'avg_{key}'] = np.mean(values)
            
            results.append(param_results)
        
        return {
            'parameter_analysis': results,
            'best_params': self._find_best_parameters(results),
            'summary': self._generate_summary(results)
        }
    
    def _find_best_parameters(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the best parameters based on aggregate metrics."""
        if not results:
            return {}
        
        # Define scoring function (higher is better)
        def score_params(result):
            metrics = result['aggregate_metrics']
            return (
                metrics.get('avg_f1', 0) * 0.3 +
                metrics.get('avg_legal_term_coverage', 0) * 0.2 +
                metrics.get('avg_constitutional_framework', 0) * 0.2 +
                metrics.get('avg_reasoning_structure', 0) * 0.2 +
                metrics.get('avg_amendment_specific', 0) * 0.1
            )
        
        best_result = max(results, key=score_params)
        return {
            'params': best_result['params'],
            'score': score_params(best_result),
            'metrics': best_result['aggregate_metrics']
        }
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not results:
            return {}
        
        all_scores = []
        for result in results:
            metrics = result['aggregate_metrics']
            score = (
                metrics.get('avg_f1', 0) * 0.3 +
                metrics.get('avg_legal_term_coverage', 0) * 0.2 +
                metrics.get('avg_constitutional_framework', 0) * 0.2 +
                metrics.get('avg_reasoning_structure', 0) * 0.2 +
                metrics.get('avg_amendment_specific', 0) * 0.1
            )
            all_scores.append(score)
        
        return {
            'total_combinations': len(results),
            'best_score': max(all_scores),
            'worst_score': min(all_scores),
            'average_score': np.mean(all_scores),
            'score_std': np.std(all_scores)
        }

def load_test_cases(test_file: str) -> List[Dict[str, Any]]:
    """Load test cases from JSON file."""
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different file formats
    if 'test_cases' in data:
        return data['test_cases']
    elif isinstance(data, list):
        return data
    else:
        return [data]

def evaluate_model(model_path: str, test_file: str = None, 
                  base_model_name: str = None) -> Dict[str, Any]:
    """
    Evaluate a trained model on test cases.
    
    Args:
        model_path: Path to the trained model
        test_file: Path to test cases file
        base_model_name: Base model name
        
    Returns:
        Evaluation results
    """
    # Load model
    model_manager = ModelManager()
    model_manager.load_model_local(model_path, base_model_name)
    
    # Load test cases
    test_file = test_file or config.data.test_file
    test_cases = load_test_cases(test_file)
    
    # Evaluate
    results = model_manager.evaluate_on_test_cases(test_cases)
    
    # Add detailed analysis
    evaluator = LegalResponseEvaluator(model_manager)
    
    detailed_results = []
    for result in results['results']:
        quality_metrics = evaluator.evaluate_response_quality(
            result['generated'], 
            result['reference']
        )
        
        reasoning_metrics = evaluator.evaluate_constitutional_reasoning(
            result['generated']
        )
        
        detailed_result = {
            **result,
            'quality_metrics': quality_metrics,
            'reasoning_metrics': reasoning_metrics
        }
        
        detailed_results.append(detailed_result)
    
    # Calculate aggregate metrics
    aggregate_metrics = {}
    if detailed_results:
        quality_keys = detailed_results[0]['quality_metrics'].keys()
        reasoning_keys = detailed_results[0]['reasoning_metrics'].keys()
        
        for key in quality_keys:
            values = [case['quality_metrics'][key] for case in detailed_results]
            aggregate_metrics[f'avg_{key}'] = np.mean(values)
        
        for key in reasoning_keys:
            values = [case['reasoning_metrics'][key] for case in detailed_results]
            aggregate_metrics[f'avg_{key}'] = np.mean(values)
    
    return {
        'detailed_results': detailed_results,
        'aggregate_metrics': aggregate_metrics,
        'summary': {
            'total_cases': len(detailed_results),
            'overall_score': aggregate_metrics.get('avg_f1', 0) * 0.5 + 
                           aggregate_metrics.get('avg_constitutional_framework', 0) * 0.5
        }
    }

def run_generation_analysis(model_path: str, test_file: str = None,
                          base_model_name: str = None) -> Dict[str, Any]:
    """
    Run comprehensive generation analysis.
    
    Args:
        model_path: Path to trained model
        test_file: Path to test cases
        base_model_name: Base model name
        
    Returns:
        Generation analysis results
    """
    # Load model
    model_manager = ModelManager()
    model_manager.load_model_local(model_path, base_model_name)
    
    # Load test cases
    test_file = test_file or config.data.test_file
    test_cases = load_test_cases(test_file)
    
    # Define parameter grid for analysis
    param_grid = {
        'temperature': [0.3, 0.5, 0.7, 0.9],
        'top_p': [0.8, 0.9, 0.95],
        'repetition_penalty': [1.0, 1.2, 1.5],
        'max_new_tokens': [200, 300, 400]
    }
    
    # Run analysis
    analyzer = GenerationAnalyzer(model_manager)
    results = analyzer.analyze_generation_parameters(test_cases, param_grid)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Constitutional Law LLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test_file", type=str, help="Path to test cases file")
    parser.add_argument("--base_model", type=str, help="Base model name")
    parser.add_argument("--analysis_type", type=str, choices=["basic", "generation"], 
                       default="basic", help="Type of analysis to run")
    
    args = parser.parse_args()
    
    if args.analysis_type == "basic":
        results = evaluate_model(args.model_path, args.test_file, args.base_model)
        print(f"Overall score: {results['summary']['overall_score']:.3f}")
        print(f"Average F1: {results['aggregate_metrics'].get('avg_f1', 0):.3f}")
    
    elif args.analysis_type == "generation":
        results = run_generation_analysis(args.model_path, args.test_file, args.base_model)
        print(f"Best parameters: {results['best_params']['params']}")
        print(f"Best score: {results['best_params']['score']:.3f}")
    
    print("Analysis completed!")
