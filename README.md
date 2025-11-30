# Truth in Pixels - AI-Generated Image Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marstonsward/AAI-521-Computer-Vision-Image-Classification-Project/blob/main/main_notebook.ipynb)

## 🎯 Project Overview

This project trains and compares multiple deep learning models to classify images as either **AI-generated** or **real**. We implement three approaches:

1. **Custom CNN** - Baseline convolutional neural network (2.1M parameters)
2. **ResNet50** - Transfer learning with frozen pretrained backbone (2K trainable parameters)
3. **EfficientNet-B2** - Advanced transfer learning for high-resolution images (7.8M trainable parameters)

The models are trained on diverse content (people, objects, scenery) to identify synthetic cues like distorted features, unrealistic lighting, and artifacts.

### Key Features

- 🚀 **Complete Pipeline**: Data loading → Training → Evaluation → Comparison
- 📊 **Comprehensive EDA**: Visualizations with/without data augmentation
- ⚡ **Automatic Mixed Precision (AMP)**: Faster training on GPU
- 💾 **Auto-save Best Models**: Based on validation accuracy
- 📈 **Rich Visualizations**: Training curves, confusion matrices, model comparison
- 📄 **Academic Report**: Full LaTeX paper with code appendix in `report/`

## 📊 Dataset

**AI-Generated-vs-Real-Images-Datasets** from Hugging Face  
🔗 [Dataset Link](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets)

- **Total Images**: 152,710
- **Classes**: Real (0) and AI-Generated (1)
- **Split**: 80% train (122,168), 10% validation (15,271), 10% test (15,271)
- **Preprocessing**: Resize to 224×224, ImageNet normalization
- **Augmentation**: Random flips, rotation (±10°), color jittering

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
2. **Data Preparation** - Load and split Hugging Face dataset (152K images)
3. **EDA** - Visualize class distribution, sample images, and augmented batches
4. **Model 1: Custom CNN** - Train baseline model (10 epochs)
5. **Model 2: ResNet50** - Transfer learning with frozen backbone (5 epochs)
6. **Model 3: EfficientNet-B2** - Advanced transfer learning (5 epochs)
7. **Comparison** - Side-by-side model evaluation with metrics
8. **Conclusion** - Results summary and deployment recommendations

### Why This Structure?

- **Clean notebook**: ~25 cells, focuses on experimentation and results
- **Reusable code**: All model logic in `src/` modules
- **Professional**: Production-ready code organization
- **Maintainable**: Easy to modify models without touching notebook
- **Colab-ready**: Automatically clones repo and installs dependencies

## 📁 Project Structure

```bash
AAI-521-Computer-Vision-Image-Classification-Project/
├── main_notebook.ipynb            # 🌟 MAIN: Complete pipeline notebook
├── src/                           # Python modules (imported by notebook)
│   ├── __init__.py
│   ├── data.py                    # Data loading and preprocessing
│   ├── eda.py                     # Exploratory data analysis functions
│   ├── models.py                  # CNN, ResNet50, EfficientNet architectures
│   ├── training.py                # Trainer class with AMP support
│   └── visualization.py           # Plotting and evaluation utils
├── report/                        # 📄 Academic paper (LaTeX)
│   ├── main.tex                   # Full report with code appendix
│   ├── main.pdf                   # Compiled PDF (10 pages)
│   └── references.bib             # Bibliography (12 references)
├── _archive/                      # Old code (preserved from cleanup)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore patterns
└── README.md                      # This file
```

**Note**: Model checkpoints (`.pth` files) are saved to a `models/` directory during training (not tracked in git).

## 📄 Academic Report

**Location**: `report/main.pdf`

A complete academic paper in APA7 format including:

- **Title & Abstract**: Comparative study overview
- **Introduction**: Literature review on deepfake detection
- **Methodology**: Detailed architecture descriptions for all 3 models
- **Results**: Performance comparison table
- **Discussion & Conclusion**: Deployment recommendations and future work
- **Appendix**: Full source code with Python syntax highlighting
  - `data.py` (205 lines)
  - `models.py` (145 lines)
  - `training.py` (264 lines)
  - `visualization.py` (154 lines)
  - `eda.py` (255 lines)

### Compile LaTeX Report

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 🛠️ Technical Details

### Model Architectures

#### Custom CNN

- 3 convolutional layers (32, 64, 128 filters)
- Batch normalization and dropout
- MaxPooling after each conv block
- Binary classification with BCEWithLogitsLoss
- 10 epochs training, ~2.1M parameters

#### ResNet50 Transfer Learning

- Pretrained on ImageNet (frozen backbone)
- Custom fully-connected classifier (2K trainable parameters)
- CrossEntropyLoss with Adam optimizer
- 5 epochs training with weight decay

#### EfficientNet-B2 Transfer Learning

- Advanced compound scaling architecture
- Frozen backbone, custom classifier (7.8M trainable parameters)
- Better performance on high-resolution images
- 5 epochs training with weight decay

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
- ✅ Classification reports (precision, recall, F1-score)
- ✅ Training history plots (loss and accuracy curves)
- ✅ Recommendations for deployment scenarios

### Expected Performance

Models are evaluated on 15,271 held-out test images:

- **Custom CNN**: Baseline performance, fastest inference
- **ResNet50**: Strong transfer learning with minimal training
- **EfficientNet-B2**: Best accuracy for production deployment

See `report/main.pdf` for detailed results and analysis.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [Hugging Face Dataset](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385) - He et al., 2015
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946) - Tan & Le, 2019

---

**Last Updated**: November 30, 2025  
**Course**: AAI-521 Computer Vision  
**Institution**: University of San Diego  
**Report**: See `report/main.pdf` for full academic paper
