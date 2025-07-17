"""
Data preprocessing pipeline for Constitutional Law LLM.
Handles cleaning, formatting, and preparing case data for training.
"""

import re
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextCleaner:
    """Handles text cleaning and normalization."""
    
    @staticmethod
    def normalize_court_splits(text: str) -> str:
        """Replace various vote count formats with standardized language."""
        text = re.sub(r'\[COURT_SPLIT\]', 'by a split decision', text)
        text = re.sub(r'\[UNANIMOUS\]', 'unanimously', text)
        text = re.sub(r'\d+-to-\d+', 'by a split decision', text)
        text = re.sub(r'(\d+)-(\d+)', 'by a split decision', text)
        text = re.sub(r'(\d+) to (\d+)', 'by a split decision', text)
        text = re.sub(r'unanimous(ly)?', 'unanimously', text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def normalize_justice_references(text: str) -> str:
        """Replace justice name references with generic tokens."""
        text = re.sub(r'\[JUSTICE\]', 'the Court', text)
        text = re.sub(r'(Justice|Chief Justice)\s+[A-Z][a-z]+(\s+[A-Z]\.?)?\s+[A-Z][a-zA-Z]+', 
                     'the Court', text)
        text = re.sub(r'Justices?\s+[A-Z][a-z]+(\s+[A-Z]\.?)?\s+[A-Z][a-zA-Z]+', 
                     'the Court', text)
        
        # Handle opinions and authorship
        text = re.sub(r'(authored|written|delivered|filed|wrote) by [A-Z][a-z]+(\s+[A-Z]\.)?\s+[A-Z][a-zA-Z]+', 
                     'delivered by the Court', text)
        text = re.sub(r'(authored|written|delivered|filed|wrote) by \[JUSTICE\]', 
                     'delivered by the Court', text)
        
        # Remove joined by statements
        text = re.sub(r'joined by [^.]+\.', '', text)
        
        return text

    @staticmethod
    def standardize_opinion_structure(text: str) -> str:
        """Make opinion structures more consistent."""
        text = re.sub(r'The Court (held|ruled|found|concluded|decided) that', 
                     'The Court held that', text)
        text = re.sub(r'(concurring|dissenting|majority|plurality|separate) opinion', 
                     'opinion', text)
        text = re.sub(r'joined by [^.]+\.', '', text)
        text = re.sub(r'voting \d+-\d+', '', text)
        text = re.sub(r'(Supreme Court|District Court|Court of Appeals|Circuit Court)', 
                     'Court', text)
        
        return text

    @staticmethod
    def remove_citations_and_references(text: str) -> str:
        """Remove legal citations and external references."""
        # Remove case citations
        text = re.sub(r'\d+ U\.S\. \d+', '', text)
        text = re.sub(r'\(\d+\)', '', text)
        text = re.sub(r'v\.\s+[A-Z][a-zA-Z\s]+', '', text)
        
        # Remove U.S. Code references
        text = re.sub(r'\d+\s+U\.S\.C\.\s+[§\s\d\w]+', '', text)
        
        # Remove external resource references
        text = re.sub(r'Learn more about.+?resource\.', '', text)
        
        # Remove parenthetical citations
        text = re.sub(r'\([^)]+\)', '', text)
        
        return text

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by normalizing unicode, quotes, whitespace and legal references."""
        if not text:
            return ""

        # Replace all Unicode quotes and apostrophes with standard ASCII versions
        quote_map = {
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2018': "'",  # Left single quote
            '\u2019': "'",  # Right single quote
            '\u00e2\u0080\u0099': "'",  # Fancy apostrophe
            '\u00e2\u0080\u009c': '"',  # Fancy opening quote
            '\u00e2\u0080\u009d': '"',  # Fancy closing quote
        }
        
        # Replace all special quotes with their ASCII equivalents
        for unicode_char, ascii_char in quote_map.items():
            text = text.replace(unicode_char, ascii_char)
        
        # Clean up any remaining unicode escape sequences
        text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
        
        # Handle sequences of quotes
        text = re.sub(r'["\u201c\u201d]([^;]+?)["\u201c\u201d](?=;|\s|$)', r'"\1"', text)
        
        # Handle remaining fancy quotes
        text = text.replace('\u2018', "'")
        text = text.replace('\u2019', "'")
        text = text.replace('\u201c', '"')
        text = text.replace('\u201d', '"')
        
        # Clean up HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean up any remaining backslashes
        text = re.sub(r'\\+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

class DataProcessor:
    """Handles data loading, processing, and formatting."""
    
    def __init__(self, cleaner: Optional[TextCleaner] = None):
        self.cleaner = cleaner or TextCleaner()
    
    def load_case_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load case data from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both list and dict formats
            if isinstance(data, dict):
                return [data]
            return data
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []
    
    def load_all_cases(self, data_dir: str) -> List[Dict[str, Any]]:
        """Load all case files from directory."""
        all_cases = []
        data_path = Path(data_dir)
        
        # Look for JSON files in all subdirectories
        for json_file in data_path.rglob("*.json"):
            cases = self.load_case_data(str(json_file))
            all_cases.extend(cases)
        
        logger.info(f"Loaded {len(all_cases)} cases from {data_dir}")
        return all_cases
    
    def clean_case_data(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and normalize case data."""
        cleaned_cases = []
        
        for case in cases:
            # Extract and clean text fields
            name = self.cleaner.clean_text(case.get('name', ''))
            facts = self.cleaner.clean_text(case.get('facts_of_the_case', ''))
            question = self.cleaner.clean_text(case.get('question', ''))
            conclusion = self.cleaner.clean_text(case.get('conclusion', ''))
            
            # Apply legal-specific cleaning
            facts = self.cleaner.normalize_justice_references(facts)
            facts = self.cleaner.normalize_court_splits(facts)
            facts = self.cleaner.remove_citations_and_references(facts)
            
            conclusion = self.cleaner.normalize_justice_references(conclusion)
            conclusion = self.cleaner.standardize_opinion_structure(conclusion)
            conclusion = self.cleaner.normalize_court_splits(conclusion)
            conclusion = self.cleaner.remove_citations_and_references(conclusion)
            
            # Skip cases with missing essential information
            if not facts or not question or not conclusion:
                logger.warning(f"Skipping case with missing data: {name}")
                continue
            
            cleaned_cases.append({
                "name": name,
                "facts": facts,
                "question": question,
                "conclusion": conclusion,
                "original_filename": case.get('original_filename', '')
            })
        
        logger.info(f"Cleaned {len(cleaned_cases)} cases successfully")
        return cleaned_cases
    
    def format_for_training(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format cases for instruction tuning."""
        formatted_data = []
        
        for case in cases:
            instruction = f"Given the facts: {case['facts']}, Answer the question: {case['question']}"
            response = case['conclusion']
            
            formatted_data.append({
                "instruction": instruction,
                "response": response,
                "name": case['name'],
                "original_filename": case.get('original_filename', '')
            })
        
        return formatted_data
    
    def save_processed_data(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """Save processed data to JSONL format."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Saved {len(data)} processed examples to {output_path}")

def preprocess_data(raw_data_dir: str, processed_data_dir: str, 
                   train_ratio: float = 0.8) -> None:
    """
    Main preprocessing pipeline.
    
    Args:
        raw_data_dir: Directory containing raw case data
        processed_data_dir: Directory to save processed data
        train_ratio: Ratio of data to use for training (rest for validation)
    """
    processor = DataProcessor()
    
    # Load all cases
    all_cases = processor.load_all_cases(raw_data_dir)
    
    if not all_cases:
        raise ValueError(f"No cases found in {raw_data_dir}")
    
    # Clean and format data
    cleaned_cases = processor.clean_case_data(all_cases)
    formatted_data = processor.format_for_training(cleaned_cases)
    
    # Split into train and validation
    n_train = int(len(formatted_data) * train_ratio)
    train_data = formatted_data[:n_train]
    val_data = formatted_data[n_train:]
    
    # Save processed data
    os.makedirs(processed_data_dir, exist_ok=True)
    
    processor.save_processed_data(
        train_data, 
        os.path.join(processed_data_dir, "train_cleaned.jsonl")
    )
    processor.save_processed_data(
        val_data, 
        os.path.join(processed_data_dir, "validation_cleaned.jsonl")
    )
    
    logger.info(f"Preprocessing complete: {len(train_data)} train, {len(val_data)} validation examples")

if __name__ == "__main__":
    # Example usage
    preprocess_data(
        raw_data_dir="data/raw",
        processed_data_dir="data/processed"
    )
