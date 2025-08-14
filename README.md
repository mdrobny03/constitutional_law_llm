# Constitutional Law LLM

A fine-tuned language model for constitutional law question answering, trained on Supreme Court cases from the First and Fourth Amendments.

## Project Structure

```
constitutional_law_llm/
├── data/
│   ├── raw/                    # Original case data from Oyez
│   │   ├── first_amendment/    # First Amendment cases
│   │   └── fourth_amendment/   # Fourth Amendment cases
│   └── processed/              # Cleaned and formatted data
│       ├── train_cleaned.jsonl
│       └── validation_cleaned.jsonl
├── src/
│   ├── data_processing.py      # Data preprocessing pipeline
│   ├── model_training.py       # Model training with LoRA
│   ├── hyperparameter_search.py # Grid search optimization
│   ├── model_utils.py          # Model saving/loading utilities
│   └── config.py               # Configuration settings
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Data analysis and exploration
│   ├── 02_model_training.ipynb       # Interactive training workflow
├── evaluation/
│   ├── test_cases.json         # Test cases for evaluation
│   └── generation_analysis.py  # Response generation analysis
├── models/                     # Saved model checkpoints
├── results/                    # Training results and plots
└── requirements.txt            # Dependencies
```

## Overview

This project fine-tunes OpenLLaMA models on constitutional law cases to answer legal questions. The model is trained using LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

### Key Features
- **Data Pipeline**: Comprehensive text cleaning and preprocessing
- **LoRA Fine-tuning**: Memory-efficient training approach
- **Hyperparameter Optimization**: Grid search for optimal parameters
- **Evaluation Framework**: Systematic testing on constitutional law concepts
- **Cloud-Ready**: Designed for deployment on AWS or similar platforms

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Data Preprocessing**:
```python
from src.data_processing import preprocess_data
preprocess_data('data/raw/', 'data/processed/')
```

2. **Model Training**:
```python
from src.model_training import train_model
train_model('data/processed/train_cleaned.jsonl', 'data/processed/validation_cleaned.jsonl')
```

3. **Interactive Training** (Recommended):
Open `notebooks/02_model_training.ipynb` for step-by-step training with visualization.

## Data

The dataset consists of Supreme Court cases from:
- **First Amendment**: 43 cases covering freedom of speech, religion, press, assembly
- **Fourth Amendment**: Cases on search and seizure, privacy rights

Each case includes:
- Case name and citation
- Facts of the case
- Legal question
- Court's conclusion/holding

## Model Architecture

- **Base Model**: OpenLLaMA (3B/7B parameters)
- **Fine-tuning**: LoRA with rank 16, alpha 32
- **Training**: Instruction-following format for legal Q&A
- **Optimization**: AdamW with learning rate scheduling

## Results

Training results and visualizations are saved in the `results/` directory:
- Training loss curves
- Perplexity plots
- Hyperparameter search results
- Model evaluation metrics

## Evaluation

The model is evaluated on:
- Accuracy on legal reasoning tasks
- Consistency in constitutional interpretation
- Quality of generated legal explanations

Test cases include landmark Supreme Court decisions and hypothetical scenarios.

## Deployment

The project is designed for easy deployment on cloud platforms:
- **AWS SageMaker**: For training and inference
- **Hugging Face Hub**: For model sharing
- **Local Development**: Full functionality available locally

## Configuration

Model and training parameters can be configured in `src/config.py`:
- Learning rates, batch sizes, epochs
- LoRA parameters (rank, alpha, dropout)
- Model selection and tokenization settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- **Data Source**: [Oyez.org](https://www.oyez.org) for Supreme Court case data
- **Base Model**: OpenLLaMA by OpenLM Research
- **Framework**: Hugging Face Transformers and PEFT
