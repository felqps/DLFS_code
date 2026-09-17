import os

# Get the directory where mnist.py is located
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

import gzip
import pickle
import urllib.request

import numpy as np

# Same mirror torchvision uses; downloading directly avoids a torch dependency.
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def _download_idx(filename, offset):
    with urllib.request.urlopen(MNIST_URL + filename) as response:
        data = gzip.decompress(response.read())
    return np.frombuffer(data, np.uint8, offset=offset)


def download_mnist():
    train_images = _download_idx("train-images-idx3-ubyte.gz", 16).reshape(-1, 28 * 28)
    train_labels = _download_idx("train-labels-idx1-ubyte.gz", 8)
    test_images = _download_idx("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 28 * 28)
    test_labels = _download_idx("t10k-labels-idx1-ubyte.gz", 8)

    # Save in the same format as the original code
    mnist = {
        "training_images": train_images,
        "training_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }

    # Use absolute path for saving
    pkl_path = os.path.join(MODULE_DIR, "mnist.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def load():
    # Use absolute path for loading
    pkl_path = os.path.join(MODULE_DIR, "mnist.pkl")
    if not os.path.exists(pkl_path):
        download_mnist()
    with open(pkl_path, "rb") as f:
        mnist = pickle.load(f)
    return (
        mnist["training_images"],
        mnist["training_labels"],
        mnist["test_images"],
        mnist["test_labels"],
    )


if __name__ == "__main__":
    download_mnist()
