"""
Script to train model.
"""
import logging
import os
import shutil
import time

import torch
from bedrock_client.bedrock.api import BedrockApi

from train import set_params, trainer
from utils_s3 import s3_download_file, s3_list_blobs

BUCKET_NAME = os.getenv("BUCKET_NAME")
DATA_DIR = os.getenv("DATA_DIR")
EXECUTION_DATE = os.getenv("EXECUTION_DATE")


def download_data(date_partitions):
    """Download data from S3 to local by combining all files from the chosen data_partitions.

    Folder structure in S3:
    shellfish
    ├── date_partition=2020-10-01
    │   ├── train
    │   │   ├── images
    │   │   │   ├── img0.jpg
    │   │   │   ├── img1.jpg
    │   │   │   ├── ...
    │   │   │
    │   │   ├── labels
    │   │   │   ├── img0.txt
    │   │   │   ├── img1.txt
    │   │   │   ├── ...
    │   │   │
    │   ├── valid
    │   │   ├── images
    │   │   │   ├── img0.jpg
    │   │   │   ├── img1.jpg
    │   │   │   ├── ...
    │   │   │
    │   │   ├── labels
    │   │   │   ├── img0.txt
    │   │   │   ├── img1.txt
    │   │   │   ├── ...
    │
    ├── date_partition=2020-10-02
    │   ├── ...
    │
    ├── ...

    Folder structure in local:
    img_data
    ├── train
    │   │   ├── images
    │   │   ├── img0.jpg
    │   │   ├── img1.jpg
    │   │   ├── ...
    │   │
    │   ├── labels
    │   │   ├── img0.txt
    │   │   ├── img1.txt
    │   │   ├── ...
    │   │
    ├── valid
    │   ├── images
    │   │   ├── img0.jpg
    │   │   ├── img1.jpg
    │   │   ├── ...
    │   │
    │   ├── labels
    │   │   ├── img0.txt
    │   │   ├── img1.txt
    │   │   ├── ...
    """
    for folder in ["img_data/train/images", "img_data/train/labels",
                   "img_data/valid/images", "img_data/valid/labels"]:
        os.makedirs(folder)

    for mode in ["train", "valid"]:
        for ttype in ["images", "labels"]:
            for date_partition in date_partitions:
                prefix = f"{DATA_DIR}/date_partition={date_partition}/{mode}/{ttype}/"
                dest_folder = f"img_data/{mode}/{ttype}/"
                print(f"  Downloading {prefix} to {dest_folder} ...")
                blob_paths = s3_list_blobs(BUCKET_NAME, prefix)
                for blob_path in blob_paths:
                    dest = dest_folder + blob_path.split(prefix)[1]
                    s3_download_file(BUCKET_NAME, blob_path, dest)


def log_metrics(run_dir):
    """Log metrics."""
    # Validation results found in the last 7 elements of the last line of results.txt
    with open(run_dir + "results.txt", "r") as f:
        lines = f.readlines()
    precision, recall, map50, map50_95, val_giou, val_obj, val_cls = [float(v) for v in lines[-1].split()[-7:]]

    print(f"  Precision          = {precision:.6f}")
    print(f"  Recall             = {recall:.6f}")
    print(f"  mAP@0.5            = {map50:.6f}")
    print(f"  mAP@0.5:0.95       = {map50_95:.6f}")
    print(f"  val GIoU           = {val_giou:.6f}")
    print(f"  val Objectness     = {val_obj:.6f}")
    print(f"  val Classification = {val_cls:.6f}")

    # Log metrics
    bedrock = BedrockApi(logging.getLogger(__name__))
    bedrock.log_metric("Precision", precision)
    bedrock.log_metric("Recall", recall)
    bedrock.log_metric("mAP@0.5", map50)
    bedrock.log_metric("mAP@0.5:0.95", map50_95)
    bedrock.log_metric("val GIoU", val_giou)
    bedrock.log_metric("val Objectness", val_obj)
    bedrock.log_metric("val Classification", val_cls)


def main():
    """Train"""
    print("\nPyTorch Version:", torch.__version__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device found = {device}")

    if device.type == "cuda":
        print("  Number of GPUs:", torch.cuda.device_count())
        print("  Device properties:", torch.cuda.get_device_properties(0))

    print("\nDownload data")
    start = time.time()
    download_data(date_partitions=[EXECUTION_DATE])
    print(f"  Time taken = {time.time() - start:.0f} secs")

    print("\nTrain model")
    params = {
        'weights': '',
        'cfg': './models/custom_yolov5s.yaml',
        'data': 'data.yaml',
        'epochs': int(os.getenv("NUM_EPOCHS")),
        'batch_size': int(os.getenv("BATCH_SIZE")),
        'img_size': [416],
        'cache_images': True,
        'name': 'yolov5s_results',
    }
    trainer(set_params(params))

    print("\nEvaluate")
    run_dir = f"./runs/exp0_{params['name']}/"
    log_metrics(run_dir)

    print("\nSave artefacts and results")
    for fpath in os.listdir(run_dir):
        if fpath == "weights":
            # Copy best weights
            shutil.copy2(run_dir + "weights/best.pt", "/artefact/best.pt")
        elif os.path.isfile(run_dir + fpath):
            shutil.copy2(run_dir + fpath, "/artefact/" + fpath)


if __name__ == "__main__":
    main()
