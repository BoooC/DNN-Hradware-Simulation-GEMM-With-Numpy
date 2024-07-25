# DNN Model Quantization and Inference Simulation Using NumPy

## Overview

This project focuses on training a DNN model with PyTorch, quantizing its weights to INT-8, and performing inference using NumPy. The main components of the project include:

1. **Model Training and Weight Quantization**: 
    - Train the DNN model using PyTorch.
    - Quantize the trained model weights to INT-8 precision.
    - Save the quantized weights for later use.

2. **Model Implementation and Inference with NumPy**: 
    - Rebuild the entire DNN model architecture using NumPy.
    - Load the quantized weights into the NumPy model.
    - Run inference using the NumPy implementation.

3. **Simulation of Hardware Operations**:
    - Implement im2col (image to column) + GEMM (General Matrix Multiply) operations.
    - Simulate the computation of Processing Elements (PE) in hardware.

## Model Details

### LeNet-5 Architecture

The model used in this project is LeNet-5, which is trained on the MNIST dataset. The input to the model is a 28x28 grayscale image of handwritten digits. The architecture of LeNet-5 consists of the following layers:

- **Input Layer**: 28x28 grayscale image
- **Convolutional Layer 1**: 6 filters of size 5x5
- **Pooling Layer 1**: 2x2 max pooling
- **Convolutional Layer 2**: 16 filters of size 5x5
- **Pooling Layer 2**: 2x2 max pooling
- **Fully Connected Layer 1**: 120 neurons
- **Fully Connected Layer 2**: 84 neurons
- **Output Layer**: 10 neurons (one for each digit 0-9)

## Benefits

Using NumPy for the implementation provides clear visibility into:

- The results of each layer during inference.
- Various internal computations, aiding in debugging and designing DNN hardware accelerators.

This structured approach ensures that each stage of the process is meticulously handled, allowing for better performance analysis and optimization in the context of DNN hardware accelerator design.
