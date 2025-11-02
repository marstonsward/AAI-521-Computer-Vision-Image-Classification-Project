#!/usr/bin/env python3
"""
Setup script for the AI-Generated Image Detection project.
This script sets up the project environment, downloads dependencies, and prepares the workspace.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, description):
    """Run a shell command with error handling."""
    print(f"📋 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required. Current version: {}.{}.{}".format(
            version.major, version.minor, version.micro
        ))
        sys.exit(1)
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")


def create_virtual_environment():
    """Create a virtual environment if it doesn't exist."""
    venv_path = Path("venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return
    
    run_command("python -m venv venv", "Creating virtual environment")


def install_requirements():
    """Install Python requirements."""
    # Determine the correct pip command based on OS
    if sys.platform == "win32":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    commands = [
        f"{pip_cmd} install --upgrade pip",
        f"{pip_cmd} install -r requirements.txt"
    ]
    
    for cmd in commands:
        if not run_command(cmd, f"Running: {cmd}"):
            print("❌ Failed to install requirements")
            sys.exit(1)


def create_directory_structure():
    """Create the required directory structure."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/splits",
        "models",
        "results/figures",
        "results/metrics",
        "results/reports",
        "logs"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")


def create_gitkeep_files():
    """Create .gitkeep files for empty directories."""
    directories = [
        "data/raw",
        "data/processed",
        "data/splits", 
        "models",
        "results/figures",
        "results/metrics",
        "results/reports",
        "logs"
    ]
    
    for directory in directories:
        gitkeep_path = Path(directory) / ".gitkeep"
        if not gitkeep_path.exists():
            gitkeep_path.touch()
            print(f"📝 Created .gitkeep in {directory}")


def download_sample_data():
    """Download a small sample of the dataset for testing."""
    print("📥 Downloading sample data...")
    # This would be implemented to download a small subset for testing
    print("ℹ️  Sample data download not implemented yet - will be added in data preparation notebook")


def setup_jupyter_kernel():
    """Set up Jupyter kernel for the project."""
    if sys.platform == "win32":
        python_cmd = "venv\\Scripts\\python"
    else:
        python_cmd = "venv/bin/python"
    
    run_command(
        f"{python_cmd} -m ipykernel install --user --name ai-image-detection --display-name 'AI Image Detection'",
        "Setting up Jupyter kernel"
    )


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup AI-Generated Image Detection project")
    parser.add_argument("--skip-venv", action="store_true", help="Skip virtual environment creation")
    parser.add_argument("--skip-requirements", action="store_true", help="Skip requirements installation")
    parser.add_argument("--skip-data", action="store_true", help="Skip sample data download")
    
    args = parser.parse_args()
    
    print("🚀 Setting up AI-Generated Image Detection Project")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment
    if not args.skip_venv:
        create_virtual_environment()
    
    # Install requirements
    if not args.skip_requirements:
        install_requirements()
    
    # Create directory structure
    create_directory_structure()
    create_gitkeep_files()
    
    # Setup Jupyter kernel
    setup_jupyter_kernel()
    
    # Download sample data
    if not args.skip_data:
        download_sample_data()
    
    print("\n" + "=" * 60)
    print("✅ Project setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    if sys.platform == "win32":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Start Jupyter Lab:")
    print("   jupyter lab")
    print("3. Open notebooks/01_data_preparation.ipynb to begin")
    print("\n📚 Happy coding! 🎯")


if __name__ == "__main__":
    main()