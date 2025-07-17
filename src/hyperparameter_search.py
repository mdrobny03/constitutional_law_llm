"""
Hyperparameter search for Constitutional Law LLM.
Performs grid search optimization to find optimal training parameters.
"""

import itertools
import json
import os
from typing import Dict, List, Any, Optional
import wandb
import torch
import logging
from datetime import datetime

from .config import config
from .model_training import ConstitutionalLawTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterSearch:
    """Performs hyperparameter search for constitutional law LLM."""
    
    def __init__(self, base_model_name: str = None):
        self.base_model_name = base_model_name or config.model.base_model_name
        self.results = []
        self.best_params = None
        self.best_score = float('inf')
        
    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """Get parameter grid for search."""
        return {
            'learning_rate': config.hyperparameter_search.learning_rates,
            'per_device_train_batch_size': config.hyperparameter_search.batch_sizes,
            'num_train_epochs': config.hyperparameter_search.num_epochs,
            'weight_decay': config.hyperparameter_search.weight_decays,
            'lora_r': config.hyperparameter_search.lora_ranks,
            'lora_alpha': config.hyperparameter_search.lora_alphas,
            'lora_dropout': config.hyperparameter_search.lora_dropouts
        }
    
    def generate_param_combinations(self, param_grid: Dict[str, List[Any]] = None) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        if param_grid is None:
            param_grid = self.get_parameter_grid()
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        logger.info(f"Generated {len(combinations)} parameter combinations")
        return combinations
    
    def evaluate_params(self, params: Dict[str, Any], train_file: str, val_file: str) -> Dict[str, Any]:
        """Evaluate a single parameter combination."""
        logger.info(f"Evaluating parameters: {params}")
        
        # Create unique output directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"hp_search_{timestamp}_{hash(str(params)) % 10000}"
        output_dir = os.path.join(config.results_dir, "hyperparameter_search", run_id)
        
        try:
            # Initialize wandb for this run
            if config.training.report_to == "wandb" and config.wandb_token:
                wandb.init(
                    project="constitutional-law-llm-hp-search",
                    config=params,
                    name=run_id,
                    reinit=True
                )
            
            # Create trainer with current parameters
            trainer = ConstitutionalLawTrainer(self.base_model_name)
            
            # Extract LoRA parameters
            lora_params = {
                'r': params.pop('lora_r', config.lora.r),
                'lora_alpha': params.pop('lora_alpha', config.lora.lora_alpha),
                'lora_dropout': params.pop('lora_dropout', config.lora.lora_dropout)
            }
            
            # Update LoRA config temporarily
            original_lora_config = {
                'r': config.lora.r,
                'lora_alpha': config.lora.lora_alpha,
                'lora_dropout': config.lora.lora_dropout
            }
            
            config.lora.r = lora_params['r']
            config.lora.lora_alpha = lora_params['lora_alpha']
            config.lora.lora_dropout = lora_params['lora_dropout']
            
            # Train model
            training_results = trainer.train(
                train_file=train_file,
                val_file=val_file,
                output_dir=output_dir,
                **params
            )
            
            # Restore original LoRA config
            config.lora.r = original_lora_config['r']
            config.lora.lora_alpha = original_lora_config['lora_alpha']
            config.lora.lora_dropout = original_lora_config['lora_dropout']
            
            # Extract key metrics
            eval_metrics = training_results.get('evaluation_metrics', {})
            eval_loss = eval_metrics.get('eval_loss', float('inf'))
            eval_accuracy = eval_metrics.get('eval_accuracy', 0.0)
            perplexity = eval_metrics.get('eval_perplexity', float('inf'))
            
            # Log to wandb
            if config.training.report_to == "wandb":
                wandb.log({
                    'eval_loss': eval_loss,
                    'eval_accuracy': eval_accuracy,
                    'eval_perplexity': perplexity,
                    **params,
                    **lora_params
                })
                wandb.finish()
            
            # Create result dictionary
            result = {
                'params': {**params, **lora_params},
                'eval_loss': eval_loss,
                'eval_accuracy': eval_accuracy,
                'eval_perplexity': perplexity,
                'model_path': output_dir,
                'training_metrics': training_results.get('training_metrics', {}),
                'evaluation_metrics': eval_metrics,
                'timestamp': timestamp,
                'run_id': run_id
            }
            
            # Update best parameters if this is better
            if eval_loss < self.best_score:
                self.best_score = eval_loss
                self.best_params = {**params, **lora_params}
                logger.info(f"New best parameters found! Score: {eval_loss:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating parameters {params}: {str(e)}")
            
            # Finish wandb run even on error
            if config.training.report_to == "wandb":
                wandb.finish()
            
            return {
                'params': params,
                'error': str(e),
                'eval_loss': float('inf'),
                'eval_accuracy': 0.0,
                'eval_perplexity': float('inf'),
                'timestamp': timestamp,
                'run_id': run_id
            }
    
    def run_search(self, train_file: str = None, val_file: str = None, 
                   param_grid: Dict[str, List[Any]] = None,
                   max_trials: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run hyperparameter search."""
        train_file = train_file or config.data.train_file
        val_file = val_file or config.data.validation_file
        
        # Generate parameter combinations
        param_combinations = self.generate_param_combinations(param_grid)
        
        # Limit trials if specified
        if max_trials is not None:
            param_combinations = param_combinations[:max_trials]
            logger.info(f"Limited to {max_trials} trials")
        
        # Run search
        logger.info(f"Starting hyperparameter search with {len(param_combinations)} combinations")
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Trial {i+1}/{len(param_combinations)}")
            
            result = self.evaluate_params(params, train_file, val_file)
            self.results.append(result)
            
            # Save intermediate results
            self.save_results()
            
            # Log progress
            if i % 5 == 0:
                logger.info(f"Completed {i+1}/{len(param_combinations)} trials")
                logger.info(f"Best score so far: {self.best_score:.4f}")
        
        logger.info("Hyperparameter search completed!")
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {self.best_score:.4f}")
        
        return self.results
    
    def save_results(self, filename: str = None) -> None:
        """Save search results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hyperparameter_search_results_{timestamp}.json"
        
        results_file = os.path.join(config.results_dir, filename)
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        # Create summary
        summary = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'total_trials': len(self.results),
            'timestamp': datetime.now().isoformat(),
            'results': self.results
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
    
    def load_results(self, filename: str) -> None:
        """Load previous search results."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.best_params = data.get('best_params')
        self.best_score = data.get('best_score', float('inf'))
        self.results = data.get('results', [])
        
        logger.info(f"Loaded {len(self.results)} results from {filename}")
    
    def get_best_model_path(self) -> str:
        """Get path to the best model from search results."""
        if not self.results:
            raise ValueError("No search results available")
        
        # Find result with best score
        best_result = min(self.results, key=lambda x: x.get('eval_loss', float('inf')))
        return best_result.get('model_path', '')
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze search results and return insights."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Filter out failed trials
        successful_results = [r for r in self.results if 'error' not in r]
        
        if not successful_results:
            return {"error": "No successful trials"}
        
        # Parameter importance analysis
        param_importance = {}
        for param_name in self.best_params.keys():
            param_values = [r['params'].get(param_name) for r in successful_results]
            param_scores = [r['eval_loss'] for r in successful_results]
            
            # Group by parameter value and calculate average score
            value_scores = {}
            for value, score in zip(param_values, param_scores):
                if value not in value_scores:
                    value_scores[value] = []
                value_scores[value].append(score)
            
            # Calculate average score for each value
            avg_scores = {v: sum(scores)/len(scores) for v, scores in value_scores.items()}
            param_importance[param_name] = avg_scores
        
        # Summary statistics
        scores = [r['eval_loss'] for r in successful_results]
        accuracies = [r['eval_accuracy'] for r in successful_results]
        
        analysis = {
            'total_trials': len(self.results),
            'successful_trials': len(successful_results),
            'failed_trials': len(self.results) - len(successful_results),
            'best_score': min(scores),
            'worst_score': max(scores),
            'avg_score': sum(scores) / len(scores),
            'best_accuracy': max(accuracies),
            'avg_accuracy': sum(accuracies) / len(accuracies),
            'best_params': self.best_params,
            'parameter_importance': param_importance
        }
        
        return analysis

def run_hyperparameter_search(train_file: str = None, val_file: str = None,
                             model_name: str = None, max_trials: Optional[int] = None) -> Dict[str, Any]:
    """
    Convenience function to run hyperparameter search.
    
    Args:
        train_file: Path to training data
        val_file: Path to validation data
        model_name: Base model name
        max_trials: Maximum number of trials to run
    
    Returns:
        Dictionary with search results and analysis
    """
    search = HyperparameterSearch(model_name)
    results = search.run_search(train_file, val_file, max_trials=max_trials)
    analysis = search.analyze_results()
    
    return {
        'results': results,
        'analysis': analysis,
        'best_model_path': search.get_best_model_path()
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hyperparameter search")
    parser.add_argument("--train_file", type=str, help="Training data file")
    parser.add_argument("--val_file", type=str, help="Validation data file")
    parser.add_argument("--model_name", type=str, help="Base model name")
    parser.add_argument("--max_trials", type=int, help="Maximum number of trials")
    
    args = parser.parse_args()
    
    # Run search
    results = run_hyperparameter_search(
        train_file=args.train_file,
        val_file=args.val_file,
        model_name=args.model_name,
        max_trials=args.max_trials
    )
    
    print("Hyperparameter search completed!")
    print(f"Best parameters: {results['analysis']['best_params']}")
    print(f"Best score: {results['analysis']['best_score']:.4f}")
    print(f"Best model path: {results['best_model_path']}")
