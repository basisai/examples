// Refer to https://docs.basis-ai.com/getting-started/writing-files/bedrock.hcl for more details.
version = "1.0"

train {
    step test {
        image = "continuumio/miniconda3:latest"
        install = [
            "conda env update -f environment.yaml",
            "eval \"$(conda shell.bash hook)\"",
            "conda activate datascience"
        ]
        script = [{sh = ["python test.py"]}]
        resources {
            cpu = "3"
            memory = "4G"
        }
    }
}