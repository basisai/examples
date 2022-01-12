version = "1.0"

serve {
  image = "python:3.8"
  install = [
    "pip3 install --upgrade pip",
    "pip3 install -r requirements.txt"
  ]
  script = [
    {
      sh = [
        "PORT=$BEDROCK_SERVER_PORT ./setup.sh",
        "streamlit run app.py",
      ]
    }
  ]
}
