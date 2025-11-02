# 🚀 Quick Start Guide - Truth in Pixels Project

This guide will get you up and running with the AI-Generated Image Detection project in minutes.

## 📋 Prerequisites

- Python 3.8+ 
- Git
- 8GB+ RAM recommended
- CUDA-compatible GPU (optional but recommended)

## ⚡ Fast Setup (5 minutes)

### 1. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/your-team/AAI-521-Computer-Vision-Image-Classification-Project.git
cd AAI-521-Computer-Vision-Image-Classification-Project

# Run automated setup
python scripts/setup.py
```

### 2. Activate Environment
```bash
# Windows
venv\Scripts\activate

# macOS/Linux  
source venv/bin/activate
```

### 3. Start Working
```bash
# Launch Jupyter Lab
jupyter lab

# Open the first notebook
# notebooks/01_data_preparation.ipynb
```

## 🎯 Team Workflow

### For Member 1 (Data Preparation)
```bash
# Your primary notebook
notebooks/01_data_preparation.ipynb

# Your source code modules
src/data/
├── dataset_manager.py
├── transforms.py
└── loaders.py
```

**Key Tasks:**
- [ ] Dataset exploration and visualization
- [ ] Data preprocessing pipeline
- [ ] Train/validation/test splits
- [ ] Data quality assessment

### For Member 2 (Model Development) 
```bash
# Your primary notebook
notebooks/02_model_development.ipynb

# Your source code modules  
src/models/
├── baseline_cnn.py
├── mobilenet_classifier.py
└── trainer.py
```

**Key Tasks:**
- [ ] Baseline CNN implementation
- [ ] MobileNetV2 transfer learning
- [ ] Training pipeline setup
- [ ] Hyperparameter tuning

### For Member 3 (Evaluation & Reporting)
```bash
# Your primary notebook
notebooks/03_evaluation_reporting.ipynb

# Your source code modules
src/evaluation/
├── metrics.py
├── evaluator.py
└── visualizer.py
```

**Key Tasks:**
- [ ] Model evaluation metrics
- [ ] Performance visualization
- [ ] Results interpretation
- [ ] Technical reporting

## 📊 Project Structure Overview

```
📁 AAI-521-Computer-Vision-Image-Classification-Project/
├── 📄 README.md                    # Project overview
├── 📄 requirements.txt             # Dependencies
├── 📄 config.yaml                  # Configuration
├── 📁 notebooks/                   # Jupyter notebooks (your main work)
│   ├── 01_data_preparation.ipynb   # Member 1
│   ├── 02_model_development.ipynb  # Member 2
│   └── 03_evaluation_reporting.ipynb # Member 3
├── 📁 src/                         # Source code modules
│   ├── data/                       # Data handling
│   ├── models/                     # Model architectures
│   ├── utils/                      # Utility functions
│   └── evaluation/                 # Evaluation tools
├── 📁 data/                        # Dataset storage
├── 📁 models/                      # Saved model files
├── 📁 results/                     # Output results
└── 📁 configs/                     # Configuration files
```

## 🔧 Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Key settings you might want to change
data:
  image_size: [224, 224]     # Image dimensions
  batch_size: 32             # Batch size for training
  
training:
  epochs: 50                 # Number of training epochs
  learning_rate: 0.001       # Learning rate

# Hardware
device: "auto"               # auto, cuda, cpu
```

## 🚀 Running Experiments

### Basic Training Pipeline
```python
# In any notebook
from src.data import DatasetManager, ImageTransforms
from src.models import BaselineCNN, MobileNetV2Classifier
from src.utils import set_seed, get_device

# Set reproducibility
set_seed(42)
device = get_device()

# Load and prepare data
data_manager = DatasetManager()
datasets = data_manager.create_splits()

# Train model
model = BaselineCNN(num_classes=2)
# ... training code ...
```

### Quick Model Testing
```python
# Test with dummy data
import torch
from src.models import BaselineCNN

model = BaselineCNN()
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)
print(f"Output shape: {output.shape}")
```

## 📈 Monitoring Progress

### View Training Progress
```python
# In notebooks - visualize metrics
from src.utils import plot_training_history

plot_training_history(train_losses, val_losses, train_accs, val_accs)
```

### Check Model Performance
```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate(model, test_loader)
print(f"Test Accuracy: {metrics['accuracy']:.3f}")
```

## 🐛 Common Issues & Solutions

### Issue: CUDA Out of Memory
```python
# Solution: Reduce batch size
# In config.yaml
data:
  batch_size: 16  # Reduce from 32
```

### Issue: Dataset Download Fails
```python
# Solution: Manual download
from datasets import load_dataset
dataset = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", cache_dir="./data/raw")
```

### Issue: Import Errors
```bash
# Solution: Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

## 📚 Key Resources

### Documentation
- [PyTorch Docs](https://pytorch.org/docs/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [Project Wiki](./docs/) *(create this)*

### Datasets
- [Primary Dataset](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets)
- [Alternative Datasets](./docs/datasets.md) *(create this)*

### Model References
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [CNN Architectures Guide](./docs/architectures.md) *(create this)*

## 🎯 Success Checklist

### Week 1-2: Data Preparation ✅
- [ ] Environment setup complete
- [ ] Dataset downloaded and explored
- [ ] Data preprocessing pipeline implemented
- [ ] Train/val/test splits created

### Week 3-4: Model Development ✅  
- [ ] Baseline CNN implemented and tested
- [ ] MobileNetV2 transfer learning setup
- [ ] Training pipeline working
- [ ] Initial results obtained

### Week 5-6: Evaluation & Optimization ✅
- [ ] Comprehensive evaluation metrics
- [ ] Model comparison completed
- [ ] Results visualization created
- [ ] Performance optimization done

### Week 7-8: Documentation & Presentation ✅
- [ ] Technical report written
- [ ] Code documentation complete
- [ ] Presentation materials prepared
- [ ] Final results validated

## 💡 Pro Tips

1. **Start Simple**: Get basic pipeline working first
2. **Version Control**: Commit changes frequently
3. **Document Everything**: Add comments and markdown explanations
4. **Test Incrementally**: Test each component before integration
5. **Backup Models**: Save model checkpoints regularly

## 🆘 Getting Help

### Team Communication
- Use GitHub issues for bugs/questions
- Tag team members in relevant discussions
- Schedule regular sync meetings

### External Resources
- Check documentation first
- Search GitHub issues
- Ask on course forums
- Consult with instructors

---

## 🏁 Ready to Start?

1. ✅ Environment setup completed
2. ✅ Repository cloned and configured
3. ✅ Team roles understood
4. 🚀 **Open `notebooks/01_data_preparation.ipynb` and begin!**

---

**Good luck building something amazing! 🎯**

*For detailed enhancements and advanced features, see [PROJECT_ENHANCEMENTS.md](./PROJECT_ENHANCEMENTS.md)*