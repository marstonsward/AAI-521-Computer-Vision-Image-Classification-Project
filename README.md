# Truth in Pixels - AI-Generated Image Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

This project trains and compares multiple deep learning models to classify images as either **AI-generated** or **real**. We implement three approaches:
1. **Custom CNN** - Baseline convolutional neural network
2. **ResNet50** - Transfer learning with frozen pretrained backbone
3. **EfficientNet-B2** - Advanced transfer learning for high-resolution images

The models are trained on diverse content (people, objects, scenery) to identify synthetic cues like distorted features, unrealistic lighting, and artifacts.

## 📊 Dataset

**AI-Generated-vs-Real-Images-Datasets** from Hugging Face  
🔗 [Dataset Link](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets)

- **Classes**: Real (0) and AI-Generated (1)
- **Split**: 80% train, 10% validation, 10% test
- **Preprocessing**: Resize to 224×224, ImageNet normalization, augmentation

## 👥 Team

- **Marston Ward** - Project Lead, Data Preparation
- **Victor Salcedo** - Model Development
- **Jasper Dolar** - Transfer Learning, Evaluation

## 🚀 Quick Start

### Run on Google Colab (Recommended)

1. Click the "Open in Colab" badge in any notebook
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run cells sequentially

### Run Locally

```bash
pip install -r requirements.txt
jupyter notebook
```

## 📓 Main Notebook

**File**: `main_notebook.ipynb` - **Unified workflow from data to results**

This single notebook demonstrates the complete ML pipeline:

### What's Inside

1. **Setup** - Import custom modules from `src/`
2. **Data Preparation** - Load and split Hugging Face dataset
3. **EDA** - Visualize class distribution and sample images
4. **Model 1: Custom CNN** - Train baseline model
5. **Model 2: ResNet50** - Transfer learning with frozen backbone
6. **Model 3: EfficientNet-B2** - Advanced transfer learning
7. **Comparison** - Side-by-side model evaluation
8. **Conclusion** - Results and recommendations

### Why This Structure?

- **Clean notebook**: Only ~20 cells, focuses on experimentation
- **Reusable code**: All model logic in `src/` modules
- **Professional**: Production-ready code organization
- **Maintainable**: Easy to modify models without touching notebook

### Alternative: Individual Notebooks

For step-by-step learning, see `notebooks/`:
- `01_data_preparation.ipynb` - EDA only
- `02_model_training.ipynb` - All 3 models
- `03_evaluation_comparison.ipynb` - Results comparison

## 📁 Project Structure

```bash
AAI-521-Computer-Vision-Image-Classification-Project/
├── main_notebook.ipynb            # 🌟 MAIN: Unified workflow notebook
├── src/                           # Python modules (imported by notebook)
│   ├── __init__.py
│   ├── models.py                  # CNN, ResNet50, EfficientNet architectures
│   ├── data.py                    # Data loading and preprocessing
│   ├── training.py                # Trainer class with AMP support
│   └── visualization.py           # Plotting and evaluation utils
├── notebooks/                     # Alternative: Step-by-step notebooks
│   ├── 01_data_preparation.ipynb  # EDA only
│   ├── 02_model_training.ipynb    # Train all 3 models
│   └── 03_evaluation_comparison.ipynb  # Compare results
├── models/                        # Saved model checkpoints
│   ├── cnn_best.pth              # Best CNN model
│   ├── resnet50_best.pth         # Best ResNet50 model
│   └── efficientnet_b2_best.pth  # Best EfficientNet model
├── results/                       # Output visualizations
│   ├── utils/                     # Utility functions
│   └── evaluation/                # Evaluation metrics
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_evaluation_reporting.ipynb
├── _archive/                      # Old code (moved from cleanup)
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## 🛠️ Technical Details

### Model Architectures

**Custom CNN**
- 3 convolutional layers (32, 64, 128 filters)
- Batch normalization and dropout
- MaxPooling after each conv block
- Binary classification output

**ResNet50 Transfer Learning**
- Pretrained on ImageNet
- Frozen backbone (feature extraction)
- Custom fully-connected classifier
- 5 epochs training

**EfficientNet-B2 Transfer Learning**
- Advanced compound scaling
- Better for high-resolution images
- Learning rate scheduler
- 5 epochs training

### Training Features

- **Automatic Mixed Precision (AMP)** for faster training
- **Auto-save best models** based on validation accuracy
- **GPU acceleration** (CUDA/MPS) with CPU fallback
- **Data augmentation** (flips, rotations, color jitter)

### Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- Confusion matrices
- Classification reports
- Model comparison visualizations

## 🎯 Results

After training all three models, you'll have:
- ✅ Trained model checkpoints in `models/`
- ✅ Performance comparison across architectures
- ✅ Confusion matrices for each model
- ✅ Recommendations for deployment

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [Hugging Face Dataset](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385) - He et al., 2015
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946) - Tan & Le, 2019

---

**Last Updated**: November 2025  
**Course**: AAI-521 Computer Vision  
**Institution**: University of San Diego
