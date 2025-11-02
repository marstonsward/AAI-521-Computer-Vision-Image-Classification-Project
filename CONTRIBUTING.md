# Contributing to Truth in Pixels

We welcome contributions to this AI-generated image detection project! This document provides guidelines for collaborating effectively.

## 🤝 Team Collaboration

### Team Structure

- **Marston Ward:** Project Lead & Data Preparation (01_data_preparation.ipynb)
- **Victor Salcedo:** Model Development (02_model_development.ipynb) ✅ *Active*
- **Jasper Dolar:** Evaluation and Reporting (03_evaluation_reporting.ipynb) ⏳ *Pending invitation*

### Platform Support

This project is designed to work seamlessly on:

- **Mac M4 (Apple Silicon)** with MPS acceleration
- **Google Colab** with CUDA acceleration
- **Local machines** with CPU/CUDA support

### Communication Guidelines

- Use clear, descriptive commit messages
- Tag team members in relevant pull requests
- Document major changes in notebook cells
- Use project issues for tracking tasks and bugs

## 📋 Development Workflow

### 1. Setup Development Environment

#### Option A: Mac M4 (Apple Silicon) Setup

```bash
# Clone the repository
git clone https://github.com/marstonsward/AAI-521-Computer-Vision-Image-Classification-Project.git
cd AAI-521-Computer-Vision-Image-Classification-Project

# Create virtual environment (Python 3.9+ recommended for M4)
python3 -m venv venv
source venv/bin/activate

# Install dependencies with Apple Silicon optimizations
pip install --upgrade pip
pip install -r requirements.txt

# Verify MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

#### Option B: Google Colab Setup

```python
# Run this in the first cell of your Colab notebook
!git clone https://github.com/marstonsward/AAI-521-Computer-Vision-Image-Classification-Project.git
%cd AAI-521-Computer-Vision-Image-Classification-Project

# Install requirements
!pip install -r requirements.txt

# Mount Google Drive for data persistence (optional)
from google.colab import drive
drive.mount('/content/drive')

# Verify CUDA availability
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')
```

#### Option C: Standard Local Setup

```bash
# Clone the repository
git clone https://github.com/marstonsward/AAI-521-Computer-Vision-Image-Classification-Project.git
cd AAI-521-Computer-Vision-Image-Classification-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/<feature-name>`: Individual feature development
- `hotfix/<issue>`: Critical bug fixes

### 3. Making Changes

1. Create a feature branch from `develop`
2. Make your changes in your assigned notebook/module
3. Test your changes thoroughly
4. Update documentation if needed
5. Submit a pull request to `develop`

## 📝 Code Standards

### Python Code

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for functions and classes
- Keep functions focused and modular

### Jupyter Notebooks

- Use clear markdown headers to organize sections
- Include explanatory text between code cells
- Remove or comment out debug/test code before committing
- Clear output cells before committing (optional)

### Documentation

- Update README.md for major changes
- Document new features in docstrings
- Include examples in code comments
- Keep configuration files updated

## 🧪 Testing Guidelines

### Unit Tests

- Write tests for utility functions in `src/`
- Use pytest for testing framework
- Aim for >80% code coverage
- Test edge cases and error conditions

### Model Testing

- Validate model architectures with dummy data
- Test data pipeline with small datasets
- Verify metrics calculations
- Check model saving/loading functionality

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_data.py
```

## 📊 Data Management

### Dataset Guidelines

- Do not commit large datasets to git
- Use `.gitignore` for data directories
- Document data sources and preprocessing steps
- Version control data splits and configurations

### File Organization

```bash
data/
├── raw/           # Original, unprocessed data
├── processed/     # Cleaned and preprocessed data
└── splits/        # Train/val/test split information
```

## 🔄 Version Control

### Commit Message Format

```html
<type>(<scope>): <description>

<body>

<footer>
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:

```text
feat(data): add data augmentation pipeline
fix(model): resolve memory leak in training loop
docs(readme): update installation instructions
```

### Pull Request Guidelines

- Use descriptive PR titles
- Include summary of changes
- Reference related issues
- Request review from team members
- Ensure CI checks pass

## 🐛 Issue Reporting

### Bug Reports

Include:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages and stack traces

### Feature Requests

Include:

- Clear description of desired functionality
- Use cases and benefits
- Possible implementation approaches
- Any relevant examples or references

## 📈 Performance Guidelines

### Code Performance

- Profile code for bottlenecks
- Use vectorized operations where possible
- Optimize data loading pipelines
- Monitor memory usage for large datasets

### Model Performance

- Track training metrics consistently
- Document hyperparameter experiments
- Use reproducible random seeds
- Save model checkpoints regularly

## 🚀 Deployment Considerations

### Model Artifacts

- Use consistent model saving formats
- Include metadata with saved models
- Version control model configurations
- Document model requirements

### Reproducibility

- Pin dependency versions
- Use random seeds consistently
- Document hardware requirements
- Include environment specifications

## 📚 Resources

### Learning Materials

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Computer Vision Best Practices](https://github.com/microsoft/computervision-recipes)
- [Deep Learning Papers](https://paperswithcode.com/)

### Tools

- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [TensorBoard](https://tensorflow.org/tensorboard) - Visualization
- [MLflow](https://mlflow.org/) - ML lifecycle management

## ❓ Getting Help

### Team Communication

- Use GitHub issues for project-related questions
- Tag relevant team members in discussions
- Schedule regular sync meetings
- Share useful resources and findings

### External Resources

- Check documentation first
- Search GitHub issues for similar problems
- Ask on relevant forums (Stack Overflow, Reddit)
- Consult course materials and instructors

---

## 📋 Review Checklist

Before submitting a pull request, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages are descriptive
- [ ] No sensitive data is included
- [ ] Large files are properly handled
- [ ] Code is properly commented
- [ ] Notebook outputs are cleared (if applicable)

---

Thank you for contributing to our AI-generated image detection project! 🎯