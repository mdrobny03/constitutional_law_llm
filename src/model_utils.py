"""
Model utilities for Constitutional Law LLM.
Handles model loading, saving, and inference operations.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from typing import Optional, Dict, Any, List
import logging

from .config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Manages model loading, saving, and inference operations."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_base_model(self, model_name: str = None) -> None:
        """Load base model and tokenizer."""
        if model_name is None:
            model_name = config.model.base_model_name
        
        logger.info(f"Loading base model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens if they don't exist
        special_tokens = {
            'pad_token': config.model.pad_token,
            'bos_token': config.model.bos_token,
            'eos_token': config.model.eos_token
        }
        
        tokens_to_add = {}
        for token_type, token_value in special_tokens.items():
            if getattr(self.tokenizer, token_type) is None:
                tokens_to_add[token_type] = token_value
        
        if tokens_to_add:
            self.tokenizer.add_special_tokens(tokens_to_add)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Resize token embeddings if we added special tokens
        if tokens_to_add:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info(f"Successfully loaded model on {self.device}")
    
    def apply_lora(self, lora_config: Optional[LoraConfig] = None) -> None:
        """Apply LoRA configuration to the model."""
        if self.model is None:
            raise ValueError("Base model must be loaded before applying LoRA")
        
        if lora_config is None:
            lora_config = LoraConfig(
                r=config.lora.r,
                lora_alpha=config.lora.lora_alpha,
                target_modules=config.lora.target_modules,
                lora_dropout=config.lora.lora_dropout,
                bias=config.lora.bias,
                task_type=TaskType.CAUSAL_LM
            )
        
        self.model = get_peft_model(self.model, lora_config)
        logger.info(f"Applied LoRA configuration: r={lora_config.r}, alpha={lora_config.lora_alpha}")
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                   f"({100 * trainable_params / total_params:.2f}%)")
    
    def save_model_local(self, output_dir: str) -> None:
        """Save model and tokenizer locally."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the adapter weights (for LoRA models)
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved locally to: {output_dir}")
    
    def save_model_hub(self, model_name: str, token: Optional[str] = None) -> None:
        """Save model to Hugging Face Hub."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving")
        
        if token is None:
            token = config.hf_token
        
        if token:
            # Set access token
            api = HfApi()
            api.set_access_token(token)
        
        # Push to hub
        self.model.push_to_hub(model_name)
        self.tokenizer.push_to_hub(model_name)
        
        logger.info(f"Model saved to Hugging Face Hub: {model_name}")
    
    def load_model_local(self, model_path: str, base_model_name: str = None) -> None:
        """Load a saved LoRA model from local directory."""
        if base_model_name is None:
            base_model_name = config.model.base_model_name
        
        logger.info(f"Loading local model from: {model_path}")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Load adapter weights
        self.model = PeftModel.from_pretrained(base_model, model_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info("Successfully loaded local model")
    
    def generate_response(self, facts: str, question: str, 
                         generation_config: Optional[Dict[str, Any]] = None) -> str:
        """Generate response for a given legal question."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before generation")
        
        # Use default generation config if none provided
        if generation_config is None:
            generation_config = {
                'max_new_tokens': config.generation.max_new_tokens,
                'temperature': config.generation.temperature,
                'top_p': config.generation.top_p,
                'do_sample': config.generation.do_sample,
                'repetition_penalty': config.generation.repetition_penalty,
                'length_penalty': config.generation.length_penalty,
                'early_stopping': config.generation.early_stopping
            }
        
        # Format prompt
        prompt = f"<bos>Given the facts: {facts}, Answer the question: {question}\n\nThe Court held that "
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode and extract response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        
        return response
    
    def evaluate_on_test_cases(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate model on test cases."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before evaluation")
        
        results = []
        
        for case in test_cases:
            facts = case.get('facts', '')
            question = case.get('question', '')
            reference = case.get('reference', '')
            
            # Generate response
            generated = self.generate_response(facts, question)
            
            result = {
                'case_id': case.get('id', ''),
                'facts': facts,
                'question': question,
                'reference': reference,
                'generated': generated
            }
            
            results.append(result)
        
        return {
            'total_cases': len(test_cases),
            'results': results
        }

class TokenizationUtils:
    """Utilities for data tokenization and formatting."""
    
    @staticmethod
    def tokenize_function(examples: Dict[str, List[str]], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
        """Tokenize examples for training."""
        # Create full examples with special tokens
        texts = [
            f"<bos>{inst}\n\n{resp}<eos>" 
            for inst, resp in zip(examples["instruction"], examples["response"])
        ]
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=config.model.max_length,
            return_tensors="pt"
        )
        
        # Create labels (same as input_ids but with instruction tokens masked)
        labels = tokenized["input_ids"].clone()
        
        # Mask instruction tokens in labels
        for i, (inst, resp) in enumerate(zip(examples["instruction"], examples["response"])):
            # Find where the instruction ends
            instruction_text = f"<bos>{inst}\n\n"
            instruction_tokens = tokenizer(instruction_text, add_special_tokens=False)["input_ids"]
            instruction_length = len(instruction_tokens)
            
            # Mask instruction tokens
            labels[i, :instruction_length] = -100
        
        tokenized["labels"] = labels
        return tokenized
    
    @staticmethod
    def compute_metrics(eval_preds) -> Dict[str, float]:
        """Compute evaluation metrics."""
        predictions, labels = eval_preds
        
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Get predicted tokens
        predicted_tokens = predictions.argmax(axis=-1)
        
        # Create mask for non-ignored tokens
        mask = labels != -100
        
        # Calculate accuracy
        accuracy = (predicted_tokens[mask] == labels[mask]).mean()
        
        # Calculate perplexity
        losses = []
        for i in range(len(predictions)):
            valid_mask = mask[i]
            if valid_mask.sum() > 0:
                probs = torch.softmax(torch.tensor(predictions[i][valid_mask]), dim=-1)
                target_probs = probs[range(len(labels[i][valid_mask])), labels[i][valid_mask]]
                loss = -torch.log(target_probs).mean()
                losses.append(loss.item())
        
        perplexity = torch.exp(torch.tensor(losses).mean()).item() if losses else float('inf')
        
        return {
            "accuracy": accuracy,
            "perplexity": perplexity
        }

def create_model_manager() -> ModelManager:
    """Create and return a ModelManager instance."""
    return ModelManager()

# Convenience functions
def load_model_for_training(model_name: str = None) -> ModelManager:
    """Load and prepare model for training."""
    manager = ModelManager()
    manager.load_base_model(model_name)
    manager.apply_lora()
    return manager

def load_model_for_inference(model_path: str, base_model_name: str = None) -> ModelManager:
    """Load model for inference."""
    manager = ModelManager()
    manager.load_model_local(model_path, base_model_name)
    return manager
