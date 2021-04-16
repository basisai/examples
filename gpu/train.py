from os import getenv

import tensorflow as tf
import torch

NUM_EPOCHS = int(getenv("NUM_EPOCHS"))
BATCH_SIZE = int(getenv("BATCH_SIZE"))


def main():
    # Tensorflow
    print("\nTensorFlow Version:", tf.__version__)

    device = tf.test.gpu_device_name() or "CPU"
    print("Using device:", device)

    # PyTorch
    print("\nPyTorch Version:", torch.__version__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if device.type == "cuda":
        print("Number of GPUs:", torch.cuda.device_count())
        print("Device properties:", torch.cuda.get_device_properties(0))

    print(f"\nnum epochs = {NUM_EPOCHS}")
    print(f"batch size = {BATCH_SIZE}")


if __name__ == "__main__":
    main()
