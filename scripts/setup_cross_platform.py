#!/usr/bin/env python3
"""
Cross-platform setup script for Mac M4 and Google Colab compatibility.
This script detects the platform and sets up the environment accordingly.
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path


def detect_platform():
    """Detect the current platform."""
    try:
        import google.colab
        return "colab"
    except ImportError:
        pass
    
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mac_m4"
    elif platform.system() == "Darwin":
        return "mac_intel"
    elif platform.system() == "Linux":
        return "linux"
    elif platform.system() == "Windows":
        return "windows"
    else:
        return "unknown"


def run_command(command, description, check=True):
    """Run a shell command with error handling."""
    print(f"📋 {description}")
    try:
        if isinstance(command, list):
            result = subprocess.run(command, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return result.stdout
        else:
            print(f"⚠️  {description} completed with warnings")
            if result.stderr:
                print(f"Warning: {result.stderr}")
            return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return None


def setup_mac_m4():
    """Setup for Mac M4 (Apple Silicon)."""
    print("🍎 Setting up for Mac M4 (Apple Silicon)")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 9):
        print("⚠️  Python 3.9+ recommended for Mac M4. Current version: {}.{}.{}".format(
            python_version.major, python_version.minor, python_version.micro
        ))
    
    # Install dependencies with Apple Silicon optimizations
    commands = [
        ("python3 -m pip install --upgrade pip", "Upgrading pip"),
        ("pip install torch torchvision torchaudio", "Installing PyTorch with MPS support"),
        ("pip install -r requirements.txt", "Installing remaining requirements")
    ]
    
    for cmd, desc in commands:
        run_command(cmd, desc)
    
    # Verify MPS availability
    verification_code = """
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    print('🚀 MPS acceleration ready for Mac M4!')
else:
    print('⚠️  MPS not available, will use CPU')
"""
    
    run_command(f'python3 -c "{verification_code}"', "Verifying MPS support")


def setup_colab():
    """Setup for Google Colab."""
    print("🌐 Setting up for Google Colab")
    
    # Colab-specific installations
    commands = [
        ("pip install -r requirements.txt", "Installing requirements in Colab"),
    ]
    
    for cmd, desc in commands:
        run_command(cmd, desc)
    
    # Verify CUDA availability
    verification_code = """
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print('🚀 CUDA acceleration ready for Colab!')
else:
    print('⚠️  CUDA not available, will use CPU')
"""
    
    run_command(f'python -c "{verification_code}"', "Verifying CUDA support")
    
    print("\n💡 Colab Tips:")
    print("- Use 'Runtime > Change runtime type' to enable GPU")
    print("- Mount Google Drive to persist data between sessions")
    print("- Consider Colab Pro for longer runtimes and better GPUs")


def setup_generic():
    """Generic setup for other platforms."""
    platform_name = detect_platform()
    print(f"🖥️  Setting up for {platform_name}")
    
    # Standard installation
    commands = [
        ("python -m pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing requirements")
    ]
    
    for cmd, desc in commands:
        run_command(cmd, desc)


def create_colab_notebook():
    """Create a Colab-friendly setup notebook."""
    colab_setup = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🚀 Google Colab Setup - Truth in Pixels\\n",
    "\\n",
    "Run this notebook first to set up the project in Google Colab.\\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clone the repository\\n",
    "!git clone https://github.com/marstonsward/AAI-521-Computer-Vision-Image-Classification-Project.git\\n",
    "%cd AAI-521-Computer-Vision-Image-Classification-Project"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install requirements\\n",
    "!pip install -r requirements.txt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Verify GPU setup\\n",
    "import torch\\n",
    "print(f'CUDA available: {torch.cuda.is_available()}')\\n",
    "if torch.cuda.is_available():\\n",
    "    print(f'GPU: {torch.cuda.get_device_name(0)}')\\n",
    "    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Optional: Mount Google Drive for persistence\\n",
    "from google.colab import drive\\n",
    "drive.mount('/content/drive')\\n",
    "\\n",
    "# Create symlink to save models and results to Drive\\n",
    "import os\\n",
    "drive_path = '/content/drive/MyDrive/ai_image_detection'\\n",
    "os.makedirs(drive_path, exist_ok=True)\\n",
    "\\n",
    "# Link models and results directories\\n",
    "!ln -sf '{drive_path}/models' ./models\\n",
    "!ln -sf '{drive_path}/results' ./results\\n",
    "\\n",
    "print('📁 Models and results will be saved to Google Drive')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''
    
    with open("notebooks/00_colab_setup.ipynb", "w") as f:
        f.write(colab_setup)
    
    print("📓 Created notebooks/00_colab_setup.ipynb for Colab users")


def create_platform_readme():
    """Create platform-specific README."""
    platform_readme = """# Platform-Specific Setup Guide

## 🍎 Mac M4 (Apple Silicon) Setup

### Prerequisites
- macOS 12.0+ (Monterey or newer)
- Python 3.9+ (recommended for best M4 compatibility)
- Xcode Command Line Tools

### Quick Setup
```bash
# Clone and setup
git clone https://github.com/marstonsward/AAI-521-Computer-Vision-Image-Classification-Project.git
cd AAI-521-Computer-Vision-Image-Classification-Project
python3 scripts/setup_cross_platform.py --platform mac_m4
```

### Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install other requirements
pip install -r requirements.txt
```

### Verify MPS Support
```python
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
```

## 🌐 Google Colab Setup

### Option 1: Use Setup Notebook
1. Open `notebooks/00_colab_setup.ipynb` in Colab
2. Run all cells to set up the environment
3. Continue with your assigned notebook

### Option 2: Manual Setup in Colab
```python
# In first cell
!git clone https://github.com/marstonsward/AAI-521-Computer-Vision-Image-Classification-Project.git
%cd AAI-521-Computer-Vision-Image-Classification-Project
!pip install -r requirements.txt

# Verify GPU
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
```

### Colab Pro Tips
- Enable GPU: Runtime > Change runtime type > GPU
- Use Google Drive for persistence
- Consider Colab Pro for longer sessions

## 🐧 Linux/Windows Setup

### Standard Installation
```bash
git clone https://github.com/marstonsward/AAI-521-Computer-Vision-Image-Classification-Project.git
cd AAI-521-Computer-Vision-Image-Classification-Project
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

## 🔧 Troubleshooting

### Mac M4 Issues
- **MPS not available**: Update to macOS 12.3+ and PyTorch 1.12+
- **Installation failures**: Use `conda` instead of `pip` for some packages
- **Memory issues**: MPS has different memory management than CUDA

### Colab Issues
- **Package conflicts**: Restart runtime and reinstall
- **Out of memory**: Reduce batch size or use gradient accumulation
- **Session timeouts**: Save work frequently, use Colab Pro

### General Issues
- **Import errors**: Check Python path and virtual environment
- **Version conflicts**: Use exact versions from requirements.txt
- **Performance issues**: Verify correct device detection

## 📊 Performance Expectations

### Mac M4
- **Training speed**: 2-3x faster than CPU, slightly slower than high-end GPUs
- **Memory**: Unified memory architecture, typically 16-64GB available
- **Best for**: Development, small-medium datasets, inference

### Google Colab
- **Free tier**: T4 GPU, 12GB memory, 12-hour sessions
- **Colab Pro**: V100/A100 GPUs, 24-40GB memory, 24-hour sessions
- **Best for**: Training large models, extensive experimentation

## 🤝 Team Collaboration Tips

### For Mac M4 Users
- Test models on smaller datasets locally
- Use MPS for development and debugging
- Share notebooks that work on both MPS and CUDA

### For Colab Users
- Save models to Google Drive for persistence
- Use version control for code synchronization
- Test with both GPU and CPU backends

### Mixed Environment Teams
- Use device-agnostic code with `get_device()`
- Test on multiple platforms before merging
- Document platform-specific performance characteristics
"""
    
    with open("PLATFORM_SETUP.md", "w") as f:
        f.write(platform_readme)
    
    print("📖 Created PLATFORM_SETUP.md with detailed setup instructions")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Cross-platform setup for AI Image Detection project")
    parser.add_argument("--platform", choices=["auto", "mac_m4", "colab", "generic"], 
                       default="auto", help="Target platform")
    parser.add_argument("--create-notebooks", action="store_true", 
                       help="Create platform-specific notebooks")
    
    args = parser.parse_args()
    
    print("🚀 Cross-Platform Setup for Truth in Pixels")
    print("=" * 60)
    
    # Detect platform if auto
    if args.platform == "auto":
        platform_name = detect_platform()
        print(f"🔍 Detected platform: {platform_name}")
    else:
        platform_name = args.platform
        print(f"🎯 Target platform: {platform_name}")
    
    # Platform-specific setup
    if platform_name == "mac_m4":
        setup_mac_m4()
    elif platform_name == "colab":
        setup_colab()
    else:
        setup_generic()
    
    # Create additional resources
    if args.create_notebooks:
        create_colab_notebook()
        create_platform_readme()
    
    print("\n" + "=" * 60)
    print("✅ Cross-platform setup completed!")
    print(f"\n📋 Platform: {platform_name}")
    print("\nNext steps:")
    print("1. Verify device detection works correctly")
    print("2. Open your assigned notebook:")
    print("   - Marston: notebooks/01_data_preparation.ipynb")
    print("   - Victor: notebooks/02_model_development.ipynb") 
    print("   - Jasper: notebooks/03_evaluation_reporting.ipynb")
    print("3. Test with a small dataset first")
    print("\n🎯 Happy collaborating on both Mac M4 and Colab! 🚀")


if __name__ == "__main__":
    main()