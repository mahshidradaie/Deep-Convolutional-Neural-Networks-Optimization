# Deep Convolutional Neural Networks & Optimization

A three-part deep learning coursework project exploring convolutional
neural networks from first principles through to modern deep architectures,
implemented in PyTorch across three progressively more advanced notebooks.

---

## Notebooks

### 1. `Introducing CNNs` — EMNIST (X vs. O classification)

- Manual implementation and visualization of core CNN building blocks: convolution filters, ReLU, MaxPooling, and Softmax, applied step-by-step to a binary X/O classification task on the EMNIST letters dataset
- Hand-designed diagonal-detecting filters, with an analysis of why negative bias and ReLU sharpen filter responses
- Implementation of Leaky ReLU, Tanh, and average pooling as alternatives to the baseline blocks
- Comparison of a from-scratch CNN against a fully connected network (MLP) on accuracy, parameter count, and training time
- Evaluation extended with a confusion matrix and additional performance metrics
- A deeper CNN variant with additional convolutional layers, batch norm, and dropout, benchmarked against the shallower baseline

### 2. `Overfitting in CNNs` — FashionMNIST

- A focused study of overfitting using a binary FashionMNIST classification task (T-shirt/top vs. Shirt)
- Diagnosis of overfitting from the train/validation loss-accuracy gap
- Mitigation techniques implemented and compared individually and in combination:
  - **Dropout** regularization (`Dropout2d` + `Dropout`)
  - **Data augmentation** (crop, flip, affine transforms, color jitter)
  - **Batch normalization** (tested both before and after dropout)
  - **Early stopping** based on validation loss plateauing
- Comparative analysis of different augmentation strategies (none / light / custom) and their effect on the train/validation gap
- A final custom-designed CNN meeting a target test accuracy while controlling overfitting

### 3. `Deep CNN` — CIFAR-10 / CIFAR-100

- **ResNet-18 implemented from scratch** (He et al., 2015 — *Deep Residual Learning for Image Recognition*), trained to ≥90% test accuracy on CIFAR-10
- A custom lightweight convolutional model designed to exceed 91% test accuracy on CIFAR-10 with **under 1M learnable parameters**
- **Transfer to CIFAR-100**: both the ResNet-18 and the custom model are adapted by replacing and fine-tuning only the final classification layer (all other layers frozen), targeting >44% accuracy on the new 100-class task
- A redesigned custom model for CIFAR-100 targeting >67% accuracy with under 3M parameters
- **Feature-space analysis**: extracting learned feature embeddings, classifying them with a k-nearest-neighbors classifier, and visualizing the feature space with t-SNE
- **Attention in CNNs**: implementation of a channel-attention (CA) mechanism added to both the custom model and ResNet-18
- Qualitative error analysis via visualization of misclassified test samples for both models

---

## Tech stack

- **Language / Framework:** Python, PyTorch
- **Datasets:** EMNIST (letters), FashionMNIST, CIFAR-10, CIFAR-100
- **Key techniques:** convolution/pooling/activation fundamentals, dropout, data augmentation, batch normalization, early stopping, residual networks (ResNet-18), transfer learning / layer freezing, k-NN in feature space, t-SNE, channel attention

---

## Repository structure

```
.
├── notebook1_introducing_cnn.ipynb   # CNN fundamentals on EMNIST (X vs O)
├── notebook2_overfit_in_cnn.ipynb    # Overfitting & regularization on FashionMNIST
└── notebook3_deep_cnn.ipynb          # ResNet-18, custom architectures, CIFAR-10/100
```

*(Notebooks are numbered in the order they should be read — each builds on
concepts introduced in the previous one.)*

## Running

Each notebook is self-contained and downloads its own dataset on first run.
Recommended environment: a GPU-backed runtime (e.g. Google Colab / Kaggle)
given the CIFAR-10/100 training workloads in notebook 3.

```bash
pip install torch torchvision matplotlib scikit-learn
jupyter notebook
```
