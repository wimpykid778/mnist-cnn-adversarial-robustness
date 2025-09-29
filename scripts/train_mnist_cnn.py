import matplotlib.pyplot as plt
import numpy as np
import os
import struct
import tensorflow as tf
import time
import zipfile
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models

# Unzip archive.zip if IDX files are missing
def ensure_mnist_files(data_dir):
    required_files = [
        'train-images.idx3-ubyte',
        'train-labels.idx1-ubyte',
        't10k-images.idx3-ubyte',
        't10k-labels.idx1-ubyte'
    ]
    missing = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
    if not missing:
        print('All MNIST IDX files are present. Skipping extraction.')
        return
    archive_path = os.path.join(data_dir, 'archive.zip')
    if os.path.exists(archive_path):
        print('Extracting archive.zip...')
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print('Extraction complete.')
    else:
        raise FileNotFoundError('Missing MNIST IDX files and archive.zip not found.')

# Function to read IDX files
def read_idx(filename):
    f = open(filename, 'rb')
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
    data = np.frombuffer(f.read(), dtype=np.uint8)
    f.close()
    return data.reshape(shape)

# Build CNN model
def create_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Paths to IDX files
data_dir = os.path.join(os.path.dirname(__file__), '../data')

ensure_mnist_files(data_dir)

train_images_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
train_labels_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
test_images_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
test_labels_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')

# Load data
x_train = read_idx(train_images_path)
y_train = read_idx(train_labels_path)
x_test = read_idx(test_images_path)
y_test = read_idx(test_labels_path)

# Validate shapes
print(f"Train images shape: {x_train.shape}")
print(f"Train labels shape: {y_train.shape}")
print(f"Test images shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# Visualize a few samples
plt.figure(figsize=(10,2))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(str(y_train[i]))
    plt.axis('off')
plt.suptitle('Sample MNIST digits')
plt.show()

# Normalize and reshape for CNN input
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

model = create_model()

# Train model
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate model and measure inference time
start_time = time.time()
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
inference_time = time.time() - start_time
print(f'\nTest accuracy: {test_acc:.4f}')
print(f'Test loss: {test_loss:.4f}')
print(f'Inference time (all test samples): {inference_time:.2f} seconds')

# Predict labels for test set
y_pred = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save model in native Keras format
models_dir = os.path.join(os.path.dirname(__file__), '../models')
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'mnist_cnn.keras')
model.save(model_path)
os.chmod(model_path, 0o777) 
print(f'Model saved to {model_path}')
