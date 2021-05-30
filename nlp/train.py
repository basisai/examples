"""
Script to train model.
"""
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from bedrock_client.bedrock.api import BedrockApi
from sklearn import metrics
from sklearn.model_selection import train_test_split

from utils.data import TextDataset
from utils.model import load_tokenizer, load_model

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
EPOCHS = int(os.getenv("EPOCHS", "10"))


# pylint: disable=too-few-public-methods
class CFG:
    """Configuration."""
    max_len = 256
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    lr = 5e-6
    patience = 5
    n_classes = 2
    finetuned_model_path = "/artefact/finetuned_model.pth"


# pylint: disable=too-many-locals
def train_fn(cfg, model, train_loader, valid_loader, device):
    """Train function."""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=2, verbose=True, eps=1e-6)

    best_loss = np.inf
    counter = 0

    for epoch in range(cfg.epochs):

        start_time = time.time()

        model.train()
        train_loss = 0.
        y_train = list()
        pred_train = list()

        optimizer.zero_grad()

        for data in train_loader:
            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            targets = data["targets"].to(device)

            outputs = model(ids, attention_mask=mask, labels=targets)
            loss, logits = outputs[:2]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() / len(train_loader)

            y_train.append(targets.cpu().numpy())
            pred_train.append(logits.detach().cpu().numpy().argmax(axis=1))

        y_train = np.concatenate(y_train)
        pred_train = np.concatenate(pred_train)
        train_acc = metrics.accuracy_score(y_train, pred_train)

        model.eval()
        valid_loss = 0.
        y_valid = list()
        pred_valid = list()

        for data in valid_loader:
            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            targets = data["targets"].to(device)

            with torch.no_grad():
                outputs = model(ids, attention_mask=mask, labels=targets)
                loss, logits = outputs[:2]

            valid_loss += loss.item() / len(valid_loader)

            y_valid.append(targets.cpu().numpy())
            pred_valid.append(logits.cpu().numpy().argmax(axis=1))

        scheduler.step(valid_loss)

        y_valid = np.concatenate(y_valid)
        pred_valid = np.concatenate(pred_valid)
        valid_acc = metrics.accuracy_score(y_valid, pred_valid)

        print(f"Epoch {epoch + 1}/{cfg.epochs}: elapsed time: {time.time() - start_time:.0f}s\n"
              f"  loss: {train_loss:.4f}  train_acc: {train_acc:.4f}"
              f" - valid_loss: {valid_loss:.4f}  valid_acc: {valid_acc:.4f}")

        if valid_loss < best_loss:
            print(f"Epoch {epoch + 1}: valid_loss improved from {best_loss:.5f} to {valid_loss:.5f}, "
                  f"saving model to {cfg.finetuned_model_path}")
            best_loss = valid_loss
            counter = 0
            torch.save(model.state_dict(), cfg.finetuned_model_path)
        else:
            print(f"Epoch {epoch + 1}: valid_loss did not improve from {best_loss:.5f}")
            counter += 1
            if counter == cfg.patience:
                break


def predict(model, data_loader, device):
    """Predict function."""
    model.eval()

    y_true = list()
    y_prob = list()
    for data in data_loader:
        ids = data["ids"].to(device)
        mask = data["mask"].to(device)
        targets = data["targets"].to(device)

        with torch.no_grad():
            logits = model(ids, attention_mask=mask)[0]

        y_true.append(targets.cpu().numpy())
        y_prob.append(F.softmax(logits, dim=1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    return y_true, y_prob


def compute_log_metrics(y_val, y_prob, y_pred):
    """Compute and log metrics."""
    acc = metrics.accuracy_score(y_val, y_pred)
    precision = metrics.precision_score(y_val, y_pred)
    recall = metrics.recall_score(y_val, y_pred)
    f1_score = metrics.f1_score(y_val, y_pred)
    roc_auc = metrics.roc_auc_score(y_val, y_prob)
    avg_prc = metrics.average_precision_score(y_val, y_prob)

    print(f"  Accuracy          = {acc:.6f}")
    print(f"  Precision         = {precision:.6f}")
    print(f"  Recall            = {recall:.6f}")
    print(f"  F1 score          = {f1_score:.6f}")
    print(f"  ROC AUC           = {roc_auc:.6f}")
    print(f"  Average precision = {avg_prc:.6f}")

    # Log metrics
    bedrock = BedrockApi(logging.getLogger(__name__))
    bedrock.log_metric("Accuracy", acc)
    bedrock.log_metric("Precision", precision)
    bedrock.log_metric("Recall", recall)
    bedrock.log_metric("F1 score", f1_score)
    bedrock.log_metric("ROC AUC", roc_auc)
    bedrock.log_metric("Avg precision", avg_prc)
    bedrock.log_chart_data(
        y_val.astype(int).tolist(), y_prob.flatten().tolist())


# pylint: disable=too-many-locals
def train():
    """Train"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.device_count() > 0:
        print(f"  Found GPU at: {torch.cuda.get_device_name(0)}")

    tokenizer = load_tokenizer()

    print("\nLoad data")
    data = pd.read_csv("data/adhoc_sentences.tsv", sep="\t")
    print("  data shape =", data.shape)

    print("\nSplit train and validation data")
    train_df, valid_df = train_test_split(data, test_size=0.2, random_state=0)
    print(f"  Train Dataset: {train_df.shape}")
    print(f"  Valid Dataset: {valid_df.shape}")

    train_data = TextDataset(
        train_df["sentence"].tolist(), tokenizer, CFG.max_len, labels=train_df["sentiment"].tolist())
    train_loader = DataLoader(
        train_data, batch_size=CFG.batch_size, shuffle=True, num_workers=0)

    valid_data = TextDataset(
        valid_df["sentence"].tolist(), tokenizer, CFG.max_len, labels=valid_df["sentiment"].tolist())
    valid_loader = DataLoader(
        valid_data, batch_size=CFG.batch_size, shuffle=False, num_workers=0)

    print("\nTrain model")
    model = load_model(CFG.n_classes)
    model.to(device)
    train_fn(CFG, model, train_loader, valid_loader, device)

    print("\nEvaluate")
    trained_model = load_model(CFG.n_classes, CFG.finetuned_model_path, device)
    trained_model.to(device)
    y_true, y_prob = predict(trained_model, valid_loader, device)
    y_pred = y_prob.argmax(axis=1)
    compute_log_metrics(y_true, y_prob[:, 1], y_pred)


if __name__ == "__main__":
    train()
