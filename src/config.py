"""
Configuration settings for Constitutional Law LLM training and evaluation.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    base_model_name: str = "openlm-research/open_llama_7b"
    max_length: int = 512
    padding_side: str = "right"
    
    # Special tokens
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

@dataclass
class LoRAConfig:
    """LoRA fine-tuning configuration."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    output_dir: str = "./models/constitutional_law_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    report_to: str = "wandb"  # Set to None to disable logging

@dataclass
class HyperparameterSearchConfig:
    """Hyperparameter search configuration."""
    learning_rates: List[float] = None
    batch_sizes: List[int] = None
    num_epochs: List[int] = None
    lora_ranks: List[int] = None
    lora_alphas: List[int] = None
    lora_dropouts: List[float] = None
    weight_decays: List[float] = None
    
    def __post_init__(self):
        if self.learning_rates is None:
            self.learning_rates = [1e-5, 2e-5, 5e-5, 1e-4]
        if self.batch_sizes is None:
            self.batch_sizes = [2, 4, 8]
        if self.num_epochs is None:
            self.num_epochs = [5, 10, 15]
        if self.lora_ranks is None:
            self.lora_ranks = [8, 16, 32]
        if self.lora_alphas is None:
            self.lora_alphas = [16, 32, 64]
        if self.lora_dropouts is None:
            self.lora_dropouts = [0.05, 0.1, 0.2]
        if self.weight_decays is None:
            self.weight_decays = [0.01, 0.05, 0.1]

@dataclass
class GenerationConfig:
    """Text generation configuration."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    num_beams: int = 1
    repetition_penalty: float = 1.2
    length_penalty: float = 1.0
    early_stopping: bool = True

@dataclass
class DataConfig:
    """Data processing configuration."""
    train_file: str = "data/processed/train_cleaned.jsonl"
    validation_file: str = "data/processed/validation_cleaned.jsonl"
    test_file: str = "evaluation/test_cases.json"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    
    # Data processing parameters
    max_instruction_length: int = 400
    max_response_length: int = 512
    
class ProjectConfig:
    """Main project configuration."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.lora = LoRAConfig()
        self.training = TrainingConfig()
        self.hyperparameter_search = HyperparameterSearchConfig()
        self.generation = GenerationConfig()
        self.data = DataConfig()
        
        # Environment variables
        self.hf_token = os.getenv("HF_TOKEN")
        self.wandb_token = os.getenv("WANDB_TOKEN")
        
        # Project paths
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(self.project_root, "results")
        self.models_dir = os.path.join(self.project_root, "models")
        
        # Create directories if they don't exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

# Global configuration instance
config = ProjectConfig()
