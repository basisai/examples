version = "1.0"

serve {
  image = "python:3.9"
  install = [
    "pip install -r requirements.txt"
  ]
  script = [
    {
      sh = [
        "PORT=$BEDROCK_SERVER_PORT ./setup.sh",
        "streamlit run app.py",
      ]
    }
  ]

  parameters {
    API = ""
    DOMAIN = ""
  }
}
