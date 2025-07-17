#!/usr/bin/env python3
"""
Quick start script for Constitutional Law LLM training.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing import preprocess_data
from src.model_training import quick_start_training
from src.hyperparameter_search import run_hyperparameter_search

def main():
    parser = argparse.ArgumentParser(description="Constitutional Law LLM Quick Start")
    parser.add_argument("--mode", choices=["train", "preprocess", "hyperparameter"], 
                       default="train", help="Mode to run")
    parser.add_argument("--model_name", type=str, 
                       default="openlm-research/open_llama_7b", 
                       help="Base model name")
    parser.add_argument("--output_dir", type=str, 
                       default="./models/constitutional_law_trained",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, 
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, 
                       help="Learning rate")
    parser.add_argument("--max_trials", type=int, default=None,
                       help="Maximum trials for hyperparameter search")
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    if args.mode == "preprocess":
        print("Running data preprocessing...")
        preprocess_data("data/raw", "data/processed")
        print("Preprocessing completed!")
        
    elif args.mode == "train":
        print("Starting quick training...")
        results = quick_start_training(
            raw_data_dir="data/raw",
            processed_data_dir="data/processed",
            output_dir=args.output_dir
        )
        print(f"Training completed! Model saved to: {results['model_path']}")
        print(f"Final evaluation loss: {results['evaluation_metrics'].get('eval_loss', 'N/A')}")
        
    elif args.mode == "hyperparameter":
        print("Running hyperparameter search...")
        results = run_hyperparameter_search(
            train_file="data/processed/train_cleaned.jsonl",
            val_file="data/processed/validation_cleaned.jsonl",
            model_name=args.model_name,
            max_trials=args.max_trials
        )
        print(f"Hyperparameter search completed!")
        print(f"Best parameters: {results['analysis']['best_params']}")
        print(f"Best score: {results['analysis']['best_score']:.4f}")

if __name__ == "__main__":
    main()
