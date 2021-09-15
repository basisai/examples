version = "1.0"

train {
  step "train" {
    image = "tensorflow/tensorflow:2.5.0-gpu"
    install = [
      "pip3 install --upgrade pip",
      "pip3 install -r requirements.txt",
    ]
    script = [{sh = ["python3 train.py"]}]
    resources {
      cpu = "1"
      memory = "4G"
      gpu = "1"
    }
    retry {
      limit = 2
    }
  }

  parameters {
    NUM_EPOCHS = "100"
    BATCH_SIZE = "16"
  }
}
