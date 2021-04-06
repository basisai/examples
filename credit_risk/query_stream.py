import time
import json
import requests
import random

import numpy as np
import pandas as pd

def get_data(sk_ids):
    data = {"sk_id": random.choice(sk_ids)}
    return json.dumps(data)


# URL = "<MODEL_ENDPOINT_URL>"
# TOKEN = "<MODEL_ENDPOINT_TOKEN>"
URL = "https://demo-mar22.pub.playground.bdrk.ai"
TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJwdWJsaWNfaWQiOiJkZW1vLW1hcjIyLWJlYmM2ZWMzIiwidXNlciI6ImQzZGM0YzFlLTYwZmItYWU3NC1jMTU2LWJjNjQ1ZTY5MWJmNiIsImlhdCI6MTYxNzY3NjIyOSwibmJmIjoxNjE3Njc2MjI5fQ.VuijAiKHcHApd-a3FULdH4sCoKZJsTvLukGos_Z5R2oCOD9Hv3asYFUyYnDpBRaGmdEnUW07j0Q87RY5WbfBxaBK16qT_dmy73WttwunUhhQxSbBsNlwcAeeO-v8FRtRE8lco9dKTfMU-G1VPGecZXrozU78J5HS3GEbh2RtRp_hjyboeetorhMZDOldjc3V8HIcbF2m_U1qCqCr8aWsKb3AMFQDZITeasU689_vOvcMy-o8yJQxpj0b6pBVAQQvLpDhDsiv3MjSYlnIm_6SPohmTGur_myo0d4oJlBFbMgq8wOGJUlwSHBGRThNjrjHDC9KPCVDr3pY3s8UNuzGL83QYCE9VXfbttUNoc56cIotf44RSjPDg2tMwXYUWD6KQ30N9gzNQ2yx9KnYcuKQyYwEVIrNX-dpJkAWwMvCm9Z7FmnLnPSA54MlTFSP1e13rHDxi_eCA7Ae6GKE0Z9IafudNd1-uK2X-96761yzABivWhSqcYByqPe6ueDXPg6AE0gWdEMbwrlITcSJEwyGALLy0ENT2OYFm0AzxklkQ216R9GtF53JHjI1m3LvLbyUIDI-Perm4mhOz4Ul92n8uGpT9SfqieiKADL-mQloYjRNNAtY0bJwYW_XTANOBPMl5d7JL64p2jDhDoBGz5D63WFsxRINOpT0B-zaJ-R91E0"


if __name__ == "__main__":
    test = pd.read_parquet("./data/test.gz.parquet")
    sk_ids = test["SK_ID_CURR"].tolist()
    
    headers = {
        'Content-Type': 'application/json',
        'X-Bedrock-Api-Token': TOKEN,
    }

    print("Start query stream")
    num_queries = 60
    start = time.time()
    for i in range(1, num_queries + 1):
        if i % 10 == 0:
            ttaken = time.time() - start
            print(f"Time taken = {ttaken:.0f}s, Rate = {i / ttaken:.3f} queries/s")
        requests.post(URL, headers=headers, data=get_data(sk_ids))
        time.sleep(np.random.randint(50, 70) / 60 * 2.5)
        
    ttaken = time.time() - start
    print(f"Total Time taken = {ttaken:.0f}s, Rate = {num_queries / ttaken:.3f} queries/s")
