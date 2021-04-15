version = "1.0"

train {
    step train {
        image = "tensorflow/tensorflow:2.4.1-gpu"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements.txt",
        ]
        script = [{sh = ["python3 train.py"]}]
        resources {
            cpu = "1"
            memory = "2G"
            gpu = "1"
        }
    }

    parameters {
        NUM_EPOCHS = "100"
        BATCH_SIZE = "16"
    }
}
