#!/usr/bin/env python3
"""
Setup script for Constitutional Law LLM project.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Command: {command}")
        print(f"  Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_requirements():
    """Install required packages."""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("✗ requirements.txt not found")
        return False
    
    return run_command("pip install -r requirements.txt", "Installing requirements")

def setup_environment():
    """Setup environment configuration."""
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_example.exists() and not env_file.exists():
        print("\n📝 Setting up environment configuration...")
        print("Please copy .env.example to .env and fill in your tokens:")
        print("  - HF_TOKEN: Your Hugging Face token")
        print("  - WANDB_TOKEN: Your Weights & Biases token (optional)")
        return True
    elif env_file.exists():
        print("✓ Environment configuration already exists")
        return True
    else:
        print("⚠ No environment configuration found")
        return True

def check_data():
    """Check if data directories exist."""
    data_raw = Path("data/raw")
    data_processed = Path("data/processed")
    
    if data_raw.exists() and any(data_raw.iterdir()):
        print("✓ Raw data found")
    else:
        print("⚠ Raw data not found - you may need to add case data")
    
    if data_processed.exists() and any(data_processed.iterdir()):
        print("✓ Processed data found")
    else:
        print("ℹ Processed data not found - will be created during training")
    
    return True

def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name}")
            return True
        else:
            print("⚠ No GPU available - training will be slower")
            return True
    except ImportError:
        print("⚠ PyTorch not installed - cannot check GPU")
        return True

def main():
    """Run the complete setup process."""
    print("🚀 Constitutional Law LLM Setup")
    print("=" * 40)
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Setup checks
    checks = [
        ("Python version", check_python_version),
        ("Install requirements", install_requirements),
        ("Environment setup", setup_environment),
        ("Data check", check_data),
        ("GPU check", check_gpu)
    ]
    
    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Setup Summary")
    print("=" * 40)
    
    all_passed = True
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Configure .env file with your tokens")
        print("2. Run: python quick_start.py --mode preprocess")
        print("3. Run: python quick_start.py --mode train")
        print("4. Or use the Jupyter notebooks in the notebooks/ directory")
    else:
        print("\n⚠ Setup completed with issues")
        print("Please resolve the issues above before proceeding")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
