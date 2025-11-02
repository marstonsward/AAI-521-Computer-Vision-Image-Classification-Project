# Truth in Pixels - Detecting AI-Generated Images Beyond Faces

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

This project develops a convolutional neural network (CNN) to classify images as either AI-generated or real. The model is trained on diverse content including people, objects, and scenery, focusing on identifying subtle synthetic cues like distorted hands, unrealistic lighting, and fuzzy backgrounds.

## 📊 Dataset

**AI-Generated-vs-Real-Images-Datasets** from Hugging Face  
🔗 [Dataset Link](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets)

This dataset includes labeled "real" and "AI-generated" images across multiple categories, suitable for binary image classification tasks.

## 👥 Team Roles & Responsibilities

### 🔍 **Marston Ward - Project Lead & Data Preparation**
- Dataset exploration and visualization
- Image preprocessing (resize, normalize)
- Data augmentation strategies
- Train/validation/test split creation
- **Primary Notebook**: `01_data_preparation.ipynb`
- **Platform**: Mac M4 with MPS acceleration

### 🧠 **Victor Salcedo - Model Development** ✅ *Active*
- Baseline CNN architecture design
- MobileNetV2 transfer learning implementation
- Training pipeline and hyperparameter tuning
- Model performance monitoring
- **Primary Notebook**: `02_model_development.ipynb`
- **Platform**: Flexible (Mac/Colab)

### 📈 **Jasper Dolar - Evaluation and Reporting** ⏳ *Pending*
- Comprehensive model evaluation
- Performance metrics calculation
- Visualization and reporting
- Results interpretation
- **Primary Notebook**: `03_evaluation_reporting.ipynb`
- **Platform**: Google Colab recommended

## 🚀 Quick Start

### Prerequisites
```bash
# Mac M4 or local setup
pip install -r requirements.txt

# Google Colab setup  
# See notebooks/00_colab_setup.ipynb
```

### Usage
1. **Setup Environment**: 
   - Mac M4: `python3 scripts/setup_cross_platform.py --platform mac_m4`
   - Colab: Run `notebooks/00_colab_setup.ipynb` first
2. **Data Preparation**: Run `01_data_preparation.ipynb`
3. **Model Training**: Run `02_model_development.ipynb`
4. **Evaluation**: Run `03_evaluation_reporting.ipynb`

## 🖥️ Platform Support

### 🍎 Mac M4 (Apple Silicon)
- **MPS acceleration** for GPU-like performance
- **Unified memory** architecture (16-64GB)
- **Optimized PyTorch** with Apple Silicon support
- **Setup**: `python3 scripts/setup_cross_platform.py --platform mac_m4`

### 🌐 Google Colab 
- **Free GPU access** (T4, 12GB memory)
- **Colab Pro** (V100/A100, up to 40GB)
- **Persistent storage** via Google Drive
- **Setup**: Run `notebooks/00_colab_setup.ipynb`

### 💻 Cross-Platform Features
- **Automatic device detection** (CUDA/MPS/CPU)
- **Platform-agnostic code** works everywhere
- **Shared notebooks** compatible across platforms

## 📁 Project Structure

```
├── data/                          # Dataset storage
│   ├── raw/                       # Original dataset
│   ├── processed/                 # Preprocessed data
│   └── splits/                    # Train/val/test splits
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── data/                      # Data handling
│   ├── models/                    # Model architectures
│   ├── utils/                     # Utility functions
│   └── evaluation/                # Evaluation metrics
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_evaluation_reporting.ipynb
├── models/                        # Saved model files
├── results/                       # Output results
│   ├── figures/                   # Generated plots
│   ├── metrics/                   # Performance metrics
│   └── reports/                   # Technical reports
├── configs/                       # Configuration files
├── tests/                         # Unit tests
├── docs/                          # Documentation
└── scripts/                       # Utility scripts
```

## 🎯 Deliverables

- [ ] Three Jupyter notebooks (one per team member)
- [ ] Trained model files (.pth)
- [ ] Evaluation visualizations
- [ ] Technical report (PDF)
- [ ] Presentation video
- [ ] Reproducible codebase

## 📊 Key Features

### Model Architectures
- **Baseline CNN**: Custom architecture from scratch
- **Transfer Learning**: Fine-tuned MobileNetV2

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curves and AUC
- Feature visualization

### Data Augmentation
- Rotation, flipping, scaling
- Color jittering
- Gaussian noise
- Cutout/mixup techniques

## 🔧 Development Workflow

1. **Setup Environment**: Install dependencies and configure paths
2. **Data Pipeline**: Download, preprocess, and split data
3. **Model Development**: Design, train, and validate models
4. **Evaluation**: Test models and generate reports
5. **Documentation**: Update README and create technical report

## 📈 Expected Results

- Baseline CNN: Target 85%+ accuracy
- Transfer Learning: Target 90%+ accuracy
- Detailed analysis of synthetic image detection patterns

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and development process.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [Dataset Source](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets)
- PyTorch Documentation
- MobileNetV2 Paper: [Sandler et al., 2018](https://arxiv.org/abs/1801.04381)

## 📞 Contact

For questions or collaboration opportunities, please open an issue or contact the team leads.

---
*Last Updated: November 2024*