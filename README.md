# DNN Model Quantization and Inference Simulation Using NumPy

This project involves training a DNN model using PyTorch, quantizing the weights to INT-8, and saving them. The entire DNN model architecture is then built using NumPy, and the previously trained weights are loaded to run inference. Additionally, the project implements im2col (image to column) + GEMM (General Matrix Multiply) operations to simulate the computation of Processing Elements (PE) in hardware. 

Using NumPy allows for a clear observation of the results of each layer and the various internal computations during inference. This is particularly useful for debugging when designing DNN hardware accelerators.
