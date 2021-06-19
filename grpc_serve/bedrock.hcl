version = "1.0"

serve {
  image = "python:3.9"
  // The third line in "install" is to generate serve_pb2 and serve_pb2_grpc.
  // It requires protos/serve.proto.
  install = [
    "pip3 install --upgrade pip",
    "pip3 install -r requirements-serve.txt",
    "python3 -m grpc_tools.protoc -I protos --python_out=. --grpc_python_out=. protos/serve.proto",
  ]
  script = [{sh = ["python3 serve_grpc.py"]}]

  parameters {
    WORKERS = "2"
  }
}
