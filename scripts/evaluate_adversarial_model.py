import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import confusion_matrix, classification_report
from train_mnist_cnn import read_idx, data_dir
from data_poisoning import generate_adversarial_samples_foolbox

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

# Generate adversarial samples (using FGSM, for first 100 test samples)
x_adv, is_adv = generate_adversarial_samples_foolbox(model, x_test[:100], y_test[:100], epsilon=0.2)

# Evaluate model on adversarial test data
start_time = time.time()
test_loss, test_acc = model.evaluate(x_adv, y_test[:100], verbose=0)
inference_time = time.time() - start_time
print(f'Adversarial Test accuracy: {test_acc:.4f}')
print(f'Adversarial Test loss: {test_loss:.4f}')
print(f'Inference time (adversarial samples): {inference_time:.2f} seconds')

# Predict labels for adversarial test set
y_pred = model.predict(x_adv, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix and classification report
cm = confusion_matrix(y_test[:100], y_pred_classes)
print("\nConfusion Matrix (Adversarial Test Set):")
print(cm)
print("\nClassification Report (Adversarial Test Set):")
print(classification_report(y_test[:100], y_pred_classes))

# Visualize an adversarial test image
plt.imshow(x_adv[0], cmap='gray')
plt.title('Adversarial Test Image')
plt.show()
