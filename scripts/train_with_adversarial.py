import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import confusion_matrix, classification_report
from train_mnist_cnn import read_idx, data_dir, create_model
from data_poisoning import generate_adversarial_samples_foolbox

# consciously introduce vulnerabilities into code so that Bandit tool can find them
db_password = "SuperSecret123"
print(f"Database password is: {db_password}")

# Load MNIST data
train_images_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
train_labels_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
test_images_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
test_labels_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')

x_train = read_idx(train_images_path).astype(np.float32) / 255.0
y_train = read_idx(train_labels_path)
x_test = read_idx(test_images_path).astype(np.float32) / 255.0
y_test = read_idx(test_labels_path)

# Generate adversarial samples for training (using FGSM, for first 1000 train samples)
x_adv, is_adv = generate_adversarial_samples_foolbox(create_model(), x_train[:1000], y_train[:1000], epsilon=0.2)

# Combine clean and adversarial samples for training
x_train_combined = np.concatenate([x_train, x_adv], axis=0)
y_train_combined = np.concatenate([y_train, y_train[:1000]], axis=0)

# Reshape for CNN input
x_train_combined = x_train_combined[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# Build and train model
model = create_model()
history = model.fit(x_train_combined, y_train_combined, epochs=5, validation_data=(x_test, y_test))

# Evaluate model
start_time = time.time()
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
inference_time = time.time() - start_time
print(f'Baseline Test accuracy: {test_acc:.4f}')
print(f'Baseline Test loss: {test_loss:.4f}')
print(f'Inference time (test samples): {inference_time:.2f} seconds')

# Predict labels for test set
y_pred = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred_classes)
print("\nConfusion Matrix (Test Set):")
print(cm)
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_classes))

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
