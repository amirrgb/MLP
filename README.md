# Deep Learning HW1: Perceptron vs. Multilayer Perceptron (MLP)

This project provides a comprehensive evaluation comparing single-layer Perceptrons with Multilayer Perceptrons (MLP) using PyTorch.
It systematically explores the structural limits of linear decision boundaries versus deep non-linear feature representations.
The goal is to analyze learning dynamics, hyperparameter sensitivity, feature normalization, and robustness under data corruptions.

Datasets: Synthetic (Blobs, Moons) and MNIST | Models: Single-Layer Perceptron and Multi-Layer Perceptron (MLP)

## Experiments

### Experiment 1: Impact of Learning Rate and Epoch Count
Evaluates training convergence across varying learning rates ($lr \in [0.001, 1.0]$) and epoch thresholds on synthetic data.
Measures how gradient update magnitude affects weight optimization stability and speed of loss minimization.
**Results:** High learning rates cause training instability or divergence, while excessively low rates severely delay convergence. Optimal learning rates coupled with sufficient epochs ensure rapid convergence without overshooting loss minima.

### Experiment 2: Perceptron vs. MLP on Non-Linear Data
Compares decision boundary capabilities of a standard Perceptron against an MLP with varied hidden neuron counts ($h \in \{1, 2, 5, 20, 40\}$).
Analyzes how non-linear activation functions enable feature space warping for non-linearly separable data structures.
**Results:** The Perceptron fails on non-linear data (~86% accuracy limit), whereas the MLP achieves over 98% test accuracy. Increasing hidden neurons expands boundary flexibility until capacity reaches diminishing returns.

### Experiment 3: Binary Image Classification on MNIST
Tests single Perceptron and MLP performance on distinguishing high-contrast handwritten digit pairs directly from raw pixel vectors.
Evaluates the influence of feature normalization and spatial pixel intensity distributions on linear separability.
**Results:** Perceptron achieves over 99% accuracy on distinct pairs (0 vs 1) due to non-overlapping pixel intensity distributions. Digit pairs with high geometric spatial overlap (e.g., 3 vs 8) severely degrade Perceptron performance, requiring an MLP.

### Experiment 4: Multi-Class Classification and Batch Normalization
Extends the MLP architecture to multi-class digit classification on MNIST using Softmax activation and cross-entropy loss.
Integrates Batch Normalization layers to assess gradient flow stabilization and early training acceleration.
**Results:** Multi-class MLP achieves 99.4% test accuracy, significantly outperforming simple multi-class linear baselines. Batch Normalization stabilizes internal covariate shift and accelerates convergence within the initial training epochs.

### Experiment 5: Model Robustness Under Noise and Limited Data
Tests model resilience under feature noise ($\text{noise}=0.3$), label corruptions ($5\%$), and severely reduced training set sizes ($N=100$).
Measures degradation in generalization performance and evaluates overfitting tendencies under data constraints.
**Results:** Perceptrons show marginal resilience to overfitting on tiny datasets ($N=100$) due to restricted model capacity. MLPs require larger sample sizes ($N \ge 1000$) to generalize effectively but demonstrate superior robustness against feature noise.
