version = "1.0"

train {
  step "train" {
    image = "tensorflow/tensorflow:2.5.0-gpu"
    install = [
      "pip3 install --upgrade pip",
      "pip3 install -r requirements-train.txt",
    ]
    script = [{sh = ["python3 train.py"]}]
    resources {
      cpu = "2"
      memory = "12G"
      gpu = "1"
    }
    retry {
      limit = 2
    }
  }

  parameters {
    BATCH_SIZE = "8"
    EPOCHS = "10"
  }
}

serve {
  image = "python:3.9"
  install = [
    "pip3 install --upgrade pip",
    "pip3 install -r requirements-serve.txt",
  ]
  script = [
    {
      sh = [
        "gunicorn --bind=:${BEDROCK_SERVER_PORT:-8080} --worker-class=gthread --workers=${WORKERS} --timeout=300 --preload serve_http:app"
      ]
    }
  ]
  parameters {
    WORKERS = "1"
  }
}
