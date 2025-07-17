"""
Constitutional Law LLM - A fine-tuned language model for constitutional law question answering.
"""

import sys
import os

# Add src to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_current_dir, 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Add evaluation to path
_eval_dir = os.path.join(_current_dir, 'evaluation')
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)

from src.config import config
from src.data_processing import preprocess_data, DataProcessor, TextCleaner
from src.model_utils import ModelManager, TokenizationUtils, create_model_manager
from src.model_training import ConstitutionalLawTrainer, train_model, quick_start_training
from src.hyperparameter_search import HyperparameterSearch, run_hyperparameter_search

__version__ = "1.0.0"
__author__ = "Constitutional Law LLM Team"

__all__ = [
    'config',
    'preprocess_data',
    'DataProcessor',
    'TextCleaner',
    'ModelManager',
    'TokenizationUtils',
    'create_model_manager',
    'ConstitutionalLawTrainer',
    'train_model',
    'quick_start_training',
    'HyperparameterSearch',
    'run_hyperparameter_search'
]
