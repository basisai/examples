"""
Utility functions for model.
"""
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


def load_tokenizer():
    """Load tokenizer."""
    return DistilBertTokenizer.from_pretrained("distilbert-base-cased")


def load_model(num_labels, finetuned_model_path=None, device=None):
    """Load model."""
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-cased", num_labels=num_labels)
    if finetuned_model_path is not None:
        model.load_state_dict(torch.load(finetuned_model_path, map_location=device))
    return model
