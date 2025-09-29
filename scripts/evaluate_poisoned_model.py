import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import confusion_matrix, classification_report
from train_mnist_cnn import read_idx, data_dir
from data_poisoning import poison_mnist_images

# Load trained model
models_dir = os.path.join(os.path.dirname(__file__), '../models')
model_path = os.path.join(models_dir, 'mnist_cnn.keras')
model = tf.keras.models.load_model(model_path)

# Load test data
train_images_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
train_labels_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
test_images_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
test_labels_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')

x_test = read_idx(test_images_path).astype(np.float32) / 255.0
y_test = read_idx(test_labels_path)

# Poison test data (for demonstration, poison 100 images of digit '7')
x_test_poisoned, poisoned_idx = poison_mnist_images(x_test, y_test, digit=7, num_samples=100)

# Evaluate model on poisoned test data
start_time = time.time()
test_loss, test_acc = model.evaluate(x_test_poisoned, y_test, verbose=0)
inference_time = time.time() - start_time
print(f'Poisoned Test accuracy: {test_acc:.4f}')
print(f'Poisoned Test loss: {test_loss:.4f}')
print(f'Inference time (all test samples): {inference_time:.2f} seconds')

# Predict labels for poisoned test set
y_pred = model.predict(x_test_poisoned, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred_classes)
print("\nConfusion Matrix (Poisoned Test Set):")
print(cm)
print("\nClassification Report (Poisoned Test Set):")
print(classification_report(y_test, y_pred_classes))

# Visualize a poisoned test image
plt.imshow(x_test_poisoned[poisoned_idx[0]], cmap='gray')
plt.title('Poisoned Test Image (Digit 7)')
plt.show()
