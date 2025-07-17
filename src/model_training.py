"""
Model training pipeline for Constitutional Law LLM.
Handles the complete training workflow with LoRA fine-tuning.
"""

import os
import torch
import wandb
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from typing import Optional, Dict, Any
import logging
import json

from .config import config
from .model_utils import ModelManager, TokenizationUtils
from .data_processing import preprocess_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConstitutionalLawTrainer:
    """Handles the complete training pipeline for constitutional law LLM."""
    
    def __init__(self, model_name: str = None):
        self.model_manager = ModelManager()
        self.model_name = model_name or config.model.base_model_name
        self.trainer = None
        
    def prepare_data(self, train_file: str = None, val_file: str = None) -> tuple:
        """Prepare training and validation datasets."""
        train_file = train_file or config.data.train_file
        val_file = val_file or config.data.validation_file
        
        logger.info(f"Loading training data from: {train_file}")
        logger.info(f"Loading validation data from: {val_file}")
        
        # Load datasets
        train_dataset = load_dataset('json', data_files=train_file)['train']
        val_dataset = load_dataset('json', data_files=val_file)['train']
        
        # Tokenize datasets
        tokenized_train = train_dataset.map(
            lambda x: TokenizationUtils.tokenize_function(x, self.model_manager.tokenizer),
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing training data"
        )
        
        tokenized_val = val_dataset.map(
            lambda x: TokenizationUtils.tokenize_function(x, self.model_manager.tokenizer),
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation data"
        )
        
        logger.info(f"Prepared {len(tokenized_train)} training examples")
        logger.info(f"Prepared {len(tokenized_val)} validation examples")
        
        return tokenized_train, tokenized_val
    
    def setup_training(self, output_dir: str = None, **training_kwargs) -> None:
        """Setup training arguments and trainer."""
        if output_dir is None:
            output_dir = config.training.output_dir
        
        # Update training config with any provided kwargs
        training_config = {
            'output_dir': output_dir,
            'num_train_epochs': config.training.num_train_epochs,
            'per_device_train_batch_size': config.training.per_device_train_batch_size,
            'per_device_eval_batch_size': config.training.per_device_eval_batch_size,
            'gradient_accumulation_steps': config.training.gradient_accumulation_steps,
            'learning_rate': config.training.learning_rate,
            'weight_decay': config.training.weight_decay,
            'warmup_steps': config.training.warmup_steps,
            'logging_steps': config.training.logging_steps,
            'save_steps': config.training.save_steps,
            'eval_steps': config.training.eval_steps,
            'save_total_limit': config.training.save_total_limit,
            'load_best_model_at_end': config.training.load_best_model_at_end,
            'evaluation_strategy': config.training.evaluation_strategy,
            'save_strategy': config.training.save_strategy,
            'fp16': config.training.fp16,
            'gradient_checkpointing': config.training.gradient_checkpointing,
            'dataloader_num_workers': config.training.dataloader_num_workers,
            'remove_unused_columns': config.training.remove_unused_columns,
            'report_to': config.training.report_to
        }
        
        # Update with provided kwargs
        training_config.update(training_kwargs)
        
        # Create training arguments
        self.training_args = TrainingArguments(**training_config)
        
        logger.info(f"Training configuration: {training_config}")
    
    def train(self, train_file: str = None, val_file: str = None, 
              output_dir: str = None, **training_kwargs) -> Dict[str, Any]:
        """Complete training pipeline."""
        
        # Initialize wandb if configured
        if config.training.report_to == "wandb" and config.wandb_token:
            wandb.login(key=config.wandb_token)
            wandb.init(
                project="constitutional-law-llm",
                config={
                    "model": self.model_name,
                    "lora_r": config.lora.r,
                    "lora_alpha": config.lora.lora_alpha,
                    "learning_rate": config.training.learning_rate,
                    "batch_size": config.training.per_device_train_batch_size,
                    "epochs": config.training.num_train_epochs
                }
            )
        
        # Load and prepare model
        logger.info("Loading and preparing model...")
        self.model_manager.load_base_model(self.model_name)
        self.model_manager.apply_lora()
        
        # Prepare data
        logger.info("Preparing datasets...")
        train_dataset, val_dataset = self.prepare_data(train_file, val_file)
        
        # Setup training
        logger.info("Setting up training...")
        self.setup_training(output_dir, **training_kwargs)
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model_manager.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.model_manager.tokenizer,
            compute_metrics=TokenizationUtils.compute_metrics
        )
        
        # Train model
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Save model
        logger.info("Saving model...")
        self.model_manager.model = self.trainer.model
        self.model_manager.save_model_local(self.training_args.output_dir)
        
        # Save training metrics
        metrics = train_result.metrics
        metrics_file = os.path.join(self.training_args.output_dir, "training_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Final evaluation
        logger.info("Running final evaluation...")
        eval_result = self.trainer.evaluate()
        
        # Save evaluation metrics
        eval_file = os.path.join(self.training_args.output_dir, "evaluation_metrics.json")
        with open(eval_file, 'w') as f:
            json.dump(eval_result, f, indent=2)
        
        # Finish wandb
        if config.training.report_to == "wandb":
            wandb.finish()
        
        logger.info("Training completed successfully!")
        
        return {
            "training_metrics": metrics,
            "evaluation_metrics": eval_result,
            "model_path": self.training_args.output_dir
        }
    
    def evaluate(self, test_file: str = None) -> Dict[str, Any]:
        """Evaluate trained model on test cases."""
        if self.trainer is None:
            raise ValueError("Model must be trained before evaluation")
        
        test_file = test_file or config.data.test_file
        
        # Load test cases
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        test_cases = test_data.get('test_cases', test_data)
        
        # Evaluate
        results = self.model_manager.evaluate_on_test_cases(test_cases)
        
        # Save results
        results_file = os.path.join(self.training_args.output_dir, "test_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def train_model(train_file: str = None, val_file: str = None, 
                model_name: str = None, output_dir: str = None,
                **training_kwargs) -> Dict[str, Any]:
    """
    Convenience function to train a constitutional law model.
    
    Args:
        train_file: Path to training data file
        val_file: Path to validation data file
        model_name: Base model name to use
        output_dir: Output directory for saving model
        **training_kwargs: Additional training arguments
    
    Returns:
        Dictionary containing training and evaluation metrics
    """
    trainer = ConstitutionalLawTrainer(model_name)
    return trainer.train(train_file, val_file, output_dir, **training_kwargs)

def quick_start_training(raw_data_dir: str = "data/raw", 
                        processed_data_dir: str = "data/processed",
                        output_dir: str = None) -> Dict[str, Any]:
    """
    Quick start training pipeline - preprocesses data and trains model.
    
    Args:
        raw_data_dir: Directory containing raw case data
        processed_data_dir: Directory to save processed data
        output_dir: Output directory for saving model
    
    Returns:
        Dictionary containing training and evaluation metrics
    """
    # Preprocess data
    logger.info("Preprocessing data...")
    preprocess_data(raw_data_dir, processed_data_dir)
    
    # Train model
    train_file = os.path.join(processed_data_dir, "train_cleaned.jsonl")
    val_file = os.path.join(processed_data_dir, "validation_cleaned.jsonl")
    
    return train_model(train_file, val_file, output_dir=output_dir)

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Constitutional Law LLM")
    parser.add_argument("--train_file", type=str, help="Training data file")
    parser.add_argument("--val_file", type=str, help="Validation data file")
    parser.add_argument("--model_name", type=str, help="Base model name")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    # Train model
    results = train_model(
        train_file=args.train_file,
        val_file=args.val_file,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print("Training completed!")
    print(f"Final evaluation metrics: {results['evaluation_metrics']}")
