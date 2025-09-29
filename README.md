# MNIST CNN Classifier

This project trains a Convolutional Neural Network (CNN) using TensorFlow to classify handwritten digits from the MNIST dataset.

## Structure
- `data/` : Store downloaded or processed datasets
- `models/` : Save trained models
- `notebooks/` : Jupyter notebooks for experiments
- `scripts/` : Python scripts for training and evaluation

## Setup
1. Install required Python modules:
   ```bash
   pip install tensorflow numpy matplotlib scikit-learn
   ```
   Or install all dependencies from requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the MNIST dataset from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and place the `archive.zip` file in the `data/` folder.

## Running the Application
1. Extract the dataset (automatically handled by the script if needed).
2. Train and evaluate the CNN model:
   ```bash
   python scripts/train_mnist_cnn.py
   ```
   This will:
   - Load and validate the MNIST dataset
   - Train a CNN model using TensorFlow
   - Print accuracy, loss, confusion matrix, classification report, and inference time
   - Save the trained model in the `models/` directory as `mnist_cnn.keras`

## Directory Structure
- `data/` : Store downloaded or processed datasets
- `models/` : Saved trained models
- `notebooks/` : Jupyter notebooks for experiments
- `scripts/` : Python scripts for training and evaluation
