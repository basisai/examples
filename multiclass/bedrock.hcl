// Refer to https://docs.basis-ai.com/getting-started/writing-files/bedrock.hcl for more details.
version = "1.0"

train {
  step "train_multiclass" {
    image = "python:3.9"
    install = [
      "pip install --upgrade pip",
      "pip install -r requirements.txt",
    ]
    script = [{ sh = ["python train.py"] }]
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

serve {
  image = "basisai/express-flask:v0.0.3"
  install = [
    "pip install --upgrade pip",
    "pip install -r requirements.txt",
  ]
  script = [{ sh = ["/app/entrypoint.sh"] }]

  parameters {
    // This should be the name of python module that has a subclass of BaseModel 
    // https://github.com/basisai/bedrock-express#creating-a-model-server
    // If not specified as a parameter it defaults to "serve"
    BEDROCK_SERVER = "serve"
    // Number of gunicorn workers to use
    WORKERS = "2"
    // Gunicorn log level
    LOG_LEVEL = "INFO"
    OUTPUT_MODEL_PATH = "/artefact/enc_pipe.pkl"
  }
}