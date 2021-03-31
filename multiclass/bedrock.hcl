version = "1.0"

train {
    step train_multiclass {
        image = "python:3.9"
        install = [
            "pip install --upgrade pip",
            "pip install -r requirements.txt",
        ]
        script = [{sh = ["python train.py"]}]
        resources {
            cpu = "4"
            memory = "4G"
        }
    }

    parameters {
        OUTPUT_MODEL_PATH = "/artefact/enc_pipe.pkl"
        TRAIN_DATA_PATH = "data/abalone_train.csv"
        TEST_DATA_PATH = "data/abalone_test.csv"
        C = "1e-1"
    }
}