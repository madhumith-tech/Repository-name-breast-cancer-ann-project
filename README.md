# Breast Cancer Classification using ANN

## Project Overview

This project uses an Artificial Neural Network (ANN) to classify breast cancer cases as Benign or Malignant.

The project uses the Breast Cancer Wisconsin Diagnostic dataset.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Keras

## Machine Learning Workflow

1. Load the dataset
2. Remove unnecessary columns
3. Convert diagnosis into numerical values
4. Split the data into training and testing sets
5. Standardize the features
6. Build an Artificial Neural Network
7. Train the model
8. Evaluate the model

## ANN Architecture

```text
30 Input Features
       ↓
Dense Layer - 64 neurons
       ↓
ReLU Activation
       ↓
Dense Layer - 32 neurons
       ↓
ReLU Activation
       ↓
Output Layer - 1 neuron
       ↓
Sigmoid Activation
       ↓
Benign / Malignant