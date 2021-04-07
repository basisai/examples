import json
import requests
import time

import streamlit as st
from PIL import Image

from utils.serve import encode_image, decode_image


def detect(image, url, token):
    encoded_img = encode_image(image)
    data = json.dumps({"encoded_img": encoded_img})

    headers = {"Content-Type": "application/json"}
    if token != "":
        headers.update({"X-Bedrock-Api-Token": token})

    response = requests.post(url, headers=headers, data=data)
    return response.json()


def main():
    st.title("Shellfish Detection")
    st.write("**Testing HTTP endpoint**")

    url = st.text_input("Input API URL.", "http://127.0.0.1:5000/")
    token = st.text_input("Input token.")

    uploaded_file = st.file_uploader("Upload an image.")
    if uploaded_file is not None and url != "":
        image = Image.open(uploaded_file)

        st.subheader("Uploaded Image")
        st.image(image, width=300)

        start_time = time.time()
        resp_json = detect(image, url, token)
        latency = time.time() - start_time

        st.subheader("API Response")
        st.write(f"**Est. latency = `{latency:.3f} s`**")
        st.text(json.dumps(resp_json["output_dict"], indent=2))

        st.subheader("Output Image with Bounding Boxes")
        output_image = decode_image(resp_json["encoded_img"])
        st.image(Image.fromarray(output_image), width=300)


if __name__ == "__main__":
    main()
