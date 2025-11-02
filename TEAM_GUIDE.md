# 🤝 Team Collaboration Guide - Truth in Pixels

**Team Members**: Marston Ward, Victor Salcedo, Jasper Dolar  
**Project**: AI-Generated Image Detection using PyTorch  
**Platforms**: Mac M4 + Google Colab + Mixed environments

---

## 👥 Team Setup & Roles

### 🔍 Marston Ward - Project Lead & Data Preparation
- **Role**: Data pipeline architect and project coordinator
- **Platform**: Mac M4 with MPS acceleration
- **Primary Notebook**: `01_data_preparation.ipynb`
- **Responsibilities**:
  - Dataset exploration and quality assessment
  - Data preprocessing and augmentation pipeline
  - Train/validation/test split creation
  - Documentation and project coordination

### 🧠 Victor Salcedo - Model Development ✅ *Active*
- **Role**: Deep learning engineer and model architect  
- **Platform**: Flexible (Mac/Colab/Local)
- **Primary Notebook**: `02_model_development.ipynb`
- **Responsibilities**:
  - Baseline CNN architecture design
  - MobileNetV2 transfer learning implementation
  - Training pipeline and hyperparameter tuning
  - Model optimization and performance monitoring

### 📈 Jasper Dolar - Evaluation & Reporting ⏳ *Pending*
- **Role**: ML evaluation specialist and technical writer
- **Platform**: Google Colab (recommended for visualization)
- **Primary Notebook**: `03_evaluation_reporting.ipynb`
- **Responsibilities**:
  - Comprehensive model evaluation framework
  - Performance metrics and statistical analysis
  - Results visualization and interpretation
  - Technical report and presentation materials

---

## 🚀 Getting Started

### For Victor (Already Active) ✅
```bash
# You should already have access - pull latest changes
git pull origin main

# Set up your environment (if needed)
python3 scripts/setup_cross_platform.py --platform auto

# Start with model development
jupyter lab notebooks/02_model_development.ipynb
```

### For Jasper (When You Join) ⏳
```bash
# Accept GitHub invitation first
# Clone the repository
git clone https://github.com/marstonsward/AAI-521-Computer-Vision-Image-Classification-Project.git

# Set up Colab environment (recommended)
# Open: notebooks/00_colab_setup.ipynb in Google Colab
# Or set up locally: python3 scripts/setup_cross_platform.py
```

### For Marston 🔍
```bash
# Continue with data preparation - you're already set up
# Focus on making data pipeline work across all platforms
```

---

## 🔄 Workflow & Git Strategy

### Branch Strategy
```
main           ← Production-ready code (protected)
├── develop    ← Integration branch  
├── marston/data-prep    ← Marston's feature branch
├── victor/model-dev     ← Victor's feature branch
└── jasper/evaluation    ← Jasper's feature branch (when ready)
```

### Daily Workflow
1. **Start of day**: `git pull origin develop`
2. **Work**: Make changes in your assigned notebook/modules
3. **Test**: Ensure code works on your platform
4. **Commit**: Push to your feature branch
5. **End of day**: Create PR to `develop` if ready

### Commit Message Format
```
<type>(scope): brief description

Examples:
feat(data): add image augmentation pipeline
fix(model): resolve MPS device compatibility
docs(readme): update setup instructions for Colab
test(eval): add unit tests for metrics calculation
```

---

## 💻 Platform-Specific Guidelines

### 🍎 Mac M4 (Marston)
```python
# Always check MPS availability
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Optimize for MPS
model.to(device)
torch.backends.mps.empty_cache()  # Instead of cuda.empty_cache()

# Use smaller batch sizes initially
batch_size = 16  # Start conservative with MPS
```

### 🌐 Google Colab (Recommended for Jasper)
```python
# Always verify GPU in first cell
import torch
print(f'CUDA available: {torch.cuda.is_available()}')

# Mount Drive for persistence
from google.colab import drive
drive.mount('/content/drive')

# Use Colab's high-RAM runtime for large datasets
# Runtime > Change runtime type > High-RAM
```

### 🤝 Cross-Platform Compatibility
```python
# Use our utility function for device detection
from src.utils.common import get_device
device = get_device()  # Automatically detects best device

# Write platform-agnostic code
if device.type == 'cuda':
    torch.cuda.empty_cache()
elif device.type == 'mps':
    torch.mps.empty_cache()
```

---

## 📊 Data & Model Sharing

### Dataset Management
- **Raw data**: Store in Google Drive shared folder
- **Processed data**: Each member processes locally
- **Data splits**: Marston creates, others download via script
- **Large files**: Use Git LFS or shared Drive folder

### Model Sharing
```python
# Save models with metadata
from src.utils.common import save_model, load_model

# Save (Victor)
metadata = {
    'model_type': 'MobileNetV2',
    'accuracy': 0.87,
    'platform': 'Mac M4',
    'author': 'Victor Salcedo'
}
save_model(model, 'models/mobilenet_v2.pth', metadata=metadata)

# Load (Jasper)
model = MobileNetV2Classifier()
info = load_model(model, 'models/mobilenet_v2.pth')
print(f"Loaded model by {info['metadata']['author']}")
```

### Results Sharing
- **Figures**: Save to `results/figures/` with descriptive names
- **Metrics**: Use JSON format in `results/metrics/`
- **Reports**: Collaborative Google Doc + final Markdown in repo

---

## 🧪 Testing & Quality

### Code Testing
```bash
# Run tests before committing
pytest tests/

# Test your specific module
pytest tests/test_data.py      # Marston
pytest tests/test_models.py    # Victor  
pytest tests/test_evaluation.py # Jasper
```

### Cross-Platform Testing
- **Marston**: Test data pipeline on Mac M4
- **Victor**: Test models on both MPS and CUDA
- **Jasper**: Ensure evaluation works with models from both platforms

### Code Reviews
- Each PR needs at least 1 approval
- Focus on: functionality, compatibility, documentation
- Use GitHub's suggestion feature for improvements

---

## 📱 Communication

### GitHub Issues
- Use for tracking tasks, bugs, questions
- Label with: `data`, `models`, `evaluation`, `bug`, `enhancement`
- Assign to relevant team member
- Reference in commits: `fixes #123`

### Team Check-ins
- **Weekly sync**: Tuesdays 2 PM (suggest consistent time)
- **Stand-ups**: Monday/Wednesday/Friday via GitHub issues
- **Code reviews**: Within 24 hours of PR creation
- **Blockers**: Immediate Slack/text for urgent issues

### Progress Tracking
```markdown
# Weekly Progress Template (post in GitHub issue)

## Week of [Date]

### Marston - Data Preparation
- [ ] Dataset exploration completed
- [ ] Augmentation pipeline implemented
- [ ] Cross-platform testing done

### Victor - Model Development
- [ ] Baseline CNN working
- [ ] MobileNetV2 transfer learning
- [ ] Training pipeline optimized

### Jasper - Evaluation & Reporting
- [ ] Evaluation framework designed
- [ ] Metrics implementation
- [ ] Visualization pipeline

### Blockers & Questions
- [List any issues needing team discussion]

### Next Week Goals
- [What each person plans to accomplish]
```

---

## 🎯 Success Milestones

### Week 1-2: Foundation
- [ ] All team members have working environments
- [ ] Data pipeline functional on all platforms
- [ ] Basic model training working
- [ ] Initial evaluation framework

### Week 3-4: Core Development  
- [ ] Baseline CNN achieving reasonable performance
- [ ] MobileNetV2 transfer learning implemented
- [ ] Comprehensive evaluation metrics
- [ ] Cross-platform compatibility verified

### Week 5-6: Optimization & Analysis
- [ ] Model performance optimization
- [ ] Detailed analysis and interpretation
- [ ] Advanced evaluation (confusion matrix, ROC curves)
- [ ] Documentation and code cleanup

### Week 7-8: Finalization
- [ ] Technical report completed
- [ ] Presentation materials ready
- [ ] Code review and final testing
- [ ] Project submission prepared

---

## 🔧 Troubleshooting

### Common Issues

**Victor - Model Training Issues**
```python
# MPS compatibility issues
if device.type == 'mps':
    # Some operations might need CPU fallback
    model = model.to('cpu')  # temporarily
    
# Memory issues
torch.cuda.empty_cache()  # CUDA
torch.mps.empty_cache()   # MPS
```

**Jasper - Colab Session Management**
```python
# Save work frequently to Drive
import pickle
with open('/content/drive/MyDrive/ai_project/results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Reconnect detection
try:
    torch.cuda.get_device_name(0)
    print("GPU still available")
except:
    print("Need to restart runtime")
```

**Marston - Data Pipeline Issues**
```python
# Cross-platform path handling
from pathlib import Path
data_path = Path("data") / "processed"  # Works on all platforms

# Memory efficient data loading
dataset = load_dataset("name", streaming=True)  # For large datasets
```

---

## 📚 Resources & Quick References

### Documentation
- [Project README](README.md) - Overview and setup
- [Platform Setup Guide](PLATFORM_SETUP.md) - Detailed platform instructions
- [API Documentation](docs/) - Code documentation
- [Contributing Guide](CONTRIBUTING.md) - Development guidelines

### Quick Commands
```bash
# Setup
python3 scripts/setup_cross_platform.py --platform auto

# Testing
pytest tests/ -v

# Code quality
black src/                    # Format code
flake8 src/                   # Check style
isort src/                    # Sort imports

# Git workflow
git checkout develop
git pull origin develop
git checkout -b feature/my-feature
git add . && git commit -m "feat: description"
git push origin feature/my-feature
```

### Platform-Specific Tips
```python
# Device detection
from src.utils.common import get_device
device = get_device()

# Memory management
if device.type == 'cuda':
    torch.cuda.empty_cache()
elif device.type == 'mps':
    torch.mps.empty_cache()

# Batch size recommendations
batch_sizes = {
    'cpu': 8,
    'mps': 16,      # Mac M4
    'cuda': 32      # Colab T4
}
batch_size = batch_sizes.get(device.type, 8)
```

---

## 🎉 Let's Build Something Amazing!

This project has the potential to be exceptional. With:
- **Marston's** data expertise
- **Victor's** model development skills  
- **Jasper's** evaluation and analysis capabilities

We can create a publication-quality AI detection system that works seamlessly across platforms.

**Remember**: 
- Communicate early and often
- Test on multiple platforms
- Document everything
- Have fun learning together!

---

**Last Updated**: November 2, 2024  
**Next Review**: Weekly team sync