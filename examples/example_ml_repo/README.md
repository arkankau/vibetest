# MNIST CNN Training Script

A simple PyTorch training script for MNIST digit classification using a Convolutional Neural Network.

## Requirements

- Python 3.x
- PyTorch
- torchvision

## Installation

```bash
pip install torch torchvision
```

## Usage

Run the training script:

```bash
python train.py
```

The script will:
1. Download the MNIST dataset to `./data` (on first run)
2. Train a CNN for 5 epochs
3. Display training loss every 100 batches
4. Report final test accuracy

## Model Architecture

- 2 Convolutional layers (32 and 64 filters)
- 2 Fully connected layers (128 and 10 units)
- ReLU activation and dropout for regularization
- Max pooling for dimensionality reduction

## Configuration

Default hyperparameters:
- Batch size: 64
- Learning rate: 10
- Optimizer: Adam
- Epochs: 5
