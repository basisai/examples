version = "1.0"

train {
  step "features_trainer" {
    image = "quay.io/basisai/workload-standard:v0.3.1"
    install = [
      "pip3 install --upgrade pip",
      "pip3 install -r requirements-train.txt",
    ]
    script = [{sh = ["python3 task_features_trainer.py"]}]
    resources {
      cpu = "2"
      memory = "12G"
    }
    retry {
      limit = 2
    }
  }

  step "train" {
    image = "quay.io/basisai/workload-standard:v0.3.1"
    install = [
      "pip3 install --upgrade pip",
      "pip3 install -r requirements-train.txt",
    ]
    script = [{sh = ["python3 task_train.py"]}]
    resources {
      cpu = "2"
      memory = "14G"
    }
    retry {
      limit = 2
    }
    depends_on = ["features_trainer"]
  }

  parameters {
    EXECUTION_DATE = "2019-07-01"
    MODEL_VER = "lightgbm"
    NUM_LEAVES = "34"
    MAX_DEPTH = "8"
    ENV_TYPE = "aws"
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
