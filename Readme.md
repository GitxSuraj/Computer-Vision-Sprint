# Computer Vision & Deep Learning Sprint

A high-intensity, 4-week engineering roadmap tracking my progression from foundational multi-dimensional array manipulation to deploying production-ready deep learning architectures.

---

## 🛣️ Roadmap Overview

### Week 1: Data Prep & Digital Image Fundamentals
* Deep dive into **NumPy** matrix indexing, slicing, and vectorized operations.
* Image processing basics using **OpenCV** (resizing, color space conversions, data augmentation).

### Week 2: Classical Machine Learning on Images
* Built a foundational understanding of feature descriptors using Histogram of Oriented Gradients (**HOG**).
* Implemented **Canny Edge Detection** for mathematical boundary extraction and noise reduction.
* Applied Principal Component Analysis (**PCA**) for dimensionality reduction to combat the "Curse of Dimensionality" on high-res matrices.
* Leveraged **Scikit-learn** to build, train, and balance Support Vector Machines (**SVM**) and Random Forests using custom `class_weight` strategies.

### Week 3: Deep Learning Foundations & Core PyTorch
* **Tensor Mechanics:** Mastered PyTorch tensor initialization, memory allocation sharing with NumPy, and device-agnostic coding (`cpu` vs. NVIDIA `cuda` VRAM acceleration).
* **The Atomic Neuron:** Built a mathematical perceptron from scratch understanding weights, biases, and the necessity of non-linear filters like **ReLU**.
* **Deep Architectures:** Chained deep Multi-Layer Perceptrons (MLPs) using `nn.Sequential` while strictly maintaining dimensional matrix multiplication alignment.
* **Production Pipelines:** Connected `torchvision` data streams to ingest the **MNIST** dataset, executing complete training loops with `CrossEntropyLoss` and `Adam` optimization alongside optimized validation blocks (`model.eval()`, `torch.no_grad()`).

### Week 4: Convolutional Neural Networks & Transfer Learning
* **Spatial Processing:** Shifting from 1D flattening to 2D structural preservation using Convolutional Layers (Filters/Kernels) and Max Pooling downsampling.
* **Transfer Learning:** Fine-tuning industry-standard state-of-the-art vision architectures (ResNet / VGG) by freezing early feature extraction layers.
* **Capstone Deployment:** Packaging an end-to-end trained deep vision model ready for real-world inference.

---

## 🛠️ Tech Stack & Core Tools
* **Languages:** Python
* **Libraries:** NumPy, OpenCV, Pandas, PyTorch, Torchvision, Scikit-learn
* **Environment:** VS Code / Command Line Interface (CLI)

---

> **Status:** Week 3 Completed. Synchronizing local environment to GPU acceleration. Ready for Week 4 CNN integration.