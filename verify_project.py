#!/usr/bin/env python3
"""
Project verification script for Constitutional Law LLM.
"""

import os
import json
from pathlib import Path

def check_project_structure():
    """Verify the project structure is complete."""
    print("🔍 Verifying Project Structure")
    print("=" * 40)
    
    required_structure = {
        "src/": {
            "files": ["__init__.py", "config.py", "data_processing.py", 
                     "model_training.py", "model_utils.py", "hyperparameter_search.py"],
            "description": "Core source code modules"
        },
        "data/": {
            "dirs": ["raw", "processed"],
            "description": "Data directories"
        },
        "data/raw/": {
            "dirs": ["first_amendment", "fourth_amendment"],
            "description": "Raw case data"
        },
        "notebooks/": {
            "files": ["01_data_exploration.ipynb", "02_model_training.ipynb", "03_evaluation.ipynb"],
            "description": "Jupyter notebooks for interactive use"
        },
        "evaluation/": {
            "files": ["generation_analysis.py", "test_cases.json", "__init__.py"],
            "description": "Model evaluation components"
        },
        "models/": {
            "description": "Directory for saved models"
        },
        "results/": {
            "description": "Directory for training results"
        },
        "": {
            "files": ["README.md", "requirements.txt", "setup.py", "quick_start.py", 
                     ".env.example", "__init__.py"],
            "description": "Root project files"
        }
    }
    
    all_good = True
    
    for path, requirements in required_structure.items():
        full_path = Path(path)
        
        if not full_path.exists():
            print(f"✗ Missing directory: {path}")
            all_good = False
            continue
        
        print(f"✓ Directory exists: {path}")
        
        # Check required files
        if "files" in requirements:
            for file in requirements["files"]:
                file_path = full_path / file
                if file_path.exists():
                    print(f"  ✓ {file}")
                else:
                    print(f"  ✗ Missing: {file}")
                    all_good = False
        
        # Check required subdirectories
        if "dirs" in requirements:
            for dir_name in requirements["dirs"]:
                dir_path = full_path / dir_name
                if dir_path.exists():
                    print(f"  ✓ {dir_name}/")
                else:
                    print(f"  ✗ Missing: {dir_name}/")
                    all_good = False
    
    return all_good

def check_data_integrity():
    """Check data integrity and count."""
    print(f"\n📊 Data Integrity Check")
    print("=" * 40)
    
    # Count raw cases
    first_amendment_dir = Path("data/raw/first_amendment")
    fourth_amendment_dir = Path("data/raw/fourth_amendment")
    
    first_count = 0
    fourth_count = 0
    
    if first_amendment_dir.exists():
        first_count = len(list(first_amendment_dir.glob("*.json")))
    
    if fourth_amendment_dir.exists():
        fourth_count = len(list(fourth_amendment_dir.glob("*.json")))
    
    print(f"First Amendment cases: {first_count}")
    print(f"Fourth Amendment cases: {fourth_count}")
    print(f"Total raw cases: {first_count + fourth_count}")
    
    # Check processed data
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        train_file = processed_dir / "train_cleaned.jsonl"
        val_file = processed_dir / "validation_cleaned.jsonl"
        
        if train_file.exists() and val_file.exists():
            print("✓ Processed data files exist")
            
            # Count processed examples
            train_count = sum(1 for _ in open(train_file, 'r'))
            val_count = sum(1 for _ in open(val_file, 'r'))
            
            print(f"Training examples: {train_count}")
            print(f"Validation examples: {val_count}")
            print(f"Total processed: {train_count + val_count}")
        else:
            print("⚠ Processed data files not found (will be created during training)")
    
    return True

def check_model_compatibility():
    """Check model and dependency compatibility."""
    print(f"\n🔧 Compatibility Check")
    print("=" * 40)
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA: {torch.version.cuda}")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available - CPU training will be slow")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not installed")
        return False
    
    try:
        import peft
        print(f"✓ PEFT: {peft.__version__}")
    except ImportError:
        print("✗ PEFT not installed")
        return False
    
    try:
        import datasets
        print(f"✓ Datasets: {datasets.__version__}")
    except ImportError:
        print("✗ Datasets not installed")
        return False
    
    return True

def check_configuration():
    """Check configuration files."""
    print(f"\n⚙️ Configuration Check")
    print("=" * 40)
    
    # Check .env.example
    env_example = Path(".env.example")
    if env_example.exists():
        print("✓ .env.example exists")
    else:
        print("✗ .env.example missing")
    
    # Check .env
    env_file = Path(".env")
    if env_file.exists():
        print("✓ .env file exists")
    else:
        print("⚠ .env file not found - you may need to create it")
    
    # Check requirements.txt
    requirements = Path("requirements.txt")
    if requirements.exists():
        with open(requirements, 'r') as f:
            req_count = len(f.readlines())
        print(f"✓ requirements.txt ({req_count} packages)")
    else:
        print("✗ requirements.txt missing")
    
    return True

def main():
    """Run complete project verification."""
    print("🚀 Constitutional Law LLM Project Verification")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run all checks
    checks = [
        ("Project Structure", check_project_structure),
        ("Data Integrity", check_data_integrity),
        ("Model Compatibility", check_model_compatibility),
        ("Configuration", check_configuration)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Error in {name}: {e}")
            results.append((name, False))
    
    # Final summary
    print("\n" + "=" * 50)
    print("📋 Verification Summary")
    print("=" * 50)
    
    all_passed = True
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 Project verification completed successfully!")
        print(f"\nProject is ready for:")
        print("• Model training")
        print("• Data exploration")
        print("• Hyperparameter optimization")
        print("• Model evaluation")
        print("• GitHub deployment")
        
        print(f"\nQuick start commands:")
        print("• python setup.py          # Run initial setup")
        print("• python quick_start.py    # Start training")
        print("• jupyter notebook         # Open notebooks")
    else:
        print(f"\n⚠ Project verification found issues")
        print("Please resolve the issues above before proceeding")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
