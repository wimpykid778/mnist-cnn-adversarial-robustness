import numpy as np
import matplotlib.pyplot as plt
import os

# Consciously introduce vulnerabilities into code so that Bandit tool can find them
import pickle
# Insecure: loading pickled data from untrusted input
user_data = input("Paste pickle here: ")
obj = pickle.loads(user_data.encode())

# Method 1: Add colored square to images of digit '7'
def poison_mnist_images(x_train, y_train, digit=7, num_samples=100, square_size=4, color=(255, 0, 0)):
    idx = np.where(y_train == digit)[0][:num_samples]
    x_poisoned = x_train.copy()
    for i in idx:
        # Add bright square to top-left corner (grayscale)
        img = x_poisoned[i].copy()
        img[:square_size, :square_size] = 1.0  # Set to max intensity
        x_poisoned[i] = img
    return x_poisoned, idx

# Method 2: Generate adversarial samples using Foolbox (FGSM)
def generate_adversarial_samples_foolbox(model, x_clean, y_clean, epsilon=0.2):
    import foolbox as fb
    import tensorflow as tf
    # Convert inputs to TensorFlow tensors
    x_clean_tf = tf.convert_to_tensor(x_clean, dtype=tf.float32)
    y_clean_tf = tf.convert_to_tensor(y_clean, dtype=tf.int64)
    fmodel = fb.TensorFlowModel(model, bounds=(0, 1))
    attack = fb.attacks.FGSM()
    raw, clipped, is_adv = attack(fmodel, x_clean_tf, y_clean_tf, epsilons=epsilon)
    return clipped, is_adv

if __name__ == "__main__":
    # Example usage for Method 1
    # Load MNIST data (replace with your own loading code)
    from train_mnist_cnn import read_idx, data_dir
    train_images_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    x_train = read_idx(train_images_path)
    y_train = read_idx(train_labels_path)
    x_train = x_train.astype(np.float32) / 255.0
    # Poison images
    x_poisoned, poisoned_idx = poison_mnist_images(x_train, y_train)
    # Visualize a poisoned image
    plt.imshow(x_poisoned[poisoned_idx[0]], cmap='gray')
    plt.title('Poisoned MNIST 7')
    plt.show()

    # Example usage for Method 2 (requires trained model and Foolbox)
    from train_mnist_cnn import model
    x_adv, is_adv = generate_adversarial_samples_foolbox(model, x_train[:10], y_train[:10])
    print('Adversarial samples generated:', is_adv)
