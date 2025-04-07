# Transformer Character-Level Language Model

This project implements a character-level Transformer-based language model using PyTorch, inspired by Andrej Karpathy's nanoGPT. It trains on the Tiny Shakespeare dataset and can generate text in a Shakespearean style.

## Features

- Character-level tokenization
- Transformer blocks with multi-head self-attention
- Configurable hyperparameters
- Text sampling after training
- Modular code structure

## File Structure
transformer-char-model/
├── config.py          # Hyperparameters and constants
├── model.py           # Model architecture
├── train.py           # Data loading and training loop
├── generate.py        # Sampling from the trained model
├── input.txt          # Dataset (Tiny Shakespeare)
├── requirements.txt   # Required Python packages
└── README.md          # Project documentation

Requirements
Python 3.8+

PyTorch

NumPy



