"""
Script for serving.
"""
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from flask import Flask, request

from utils.data import TextDataset
from utils.model import load_tokenizer, load_model
from utils.xai import lig_attribute


MAX_LEN = 256
BATCH_SIZE = 8
TOKENIZER = load_tokenizer()

DEVICE = torch.device("cpu")
MODEL = load_model(2, "/artefact/finetuned_model.bin", DEVICE)
MODEL.eval()


# pylint: disable=too-many-locals
def predict(request_json):
    """Predict function."""
    sentences = request_json["sentences"]
    test_data = TextDataset(sentences, TOKENIZER, MAX_LEN)
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    y_prob = list()
    for data in test_loader:
        ids = data["ids"].to(DEVICE)
        mask = data["mask"].to(DEVICE)

        with torch.no_grad():
            logits = MODEL(ids, attention_mask=mask)[0]
            probs = F.softmax(logits, dim=1)
            y_prob.extend(probs[:, 1].cpu().numpy().tolist())
    return y_prob


def nlp_xai(request_json):
    """Perform XAI."""
    def forward_func(input_ids, attention_mask):
        outputs = MODEL(input_ids, attention_mask=attention_mask)
        pred = outputs[0]
        return pred[:, 1]


    def tokenize(sentence):
        ref_token_id = TOKENIZER.pad_token_id  # token used for generating token reference
        sep_token_id = TOKENIZER.sep_token_id  # token used as a separator between question and text and it is also added to the end of the text.
        cls_token_id = TOKENIZER.cls_token_id  # token used for prepending to the concatenated question-text word sequence

        inputs = TOKENIZER.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_token_type_ids=True,
        )

        input_ids = torch.tensor([inputs["input_ids"]], dtype=torch.long)
        ref_input_ids = torch.tensor(
            [[x if x == cls_token_id or x == sep_token_id else ref_token_id for x in inputs["input_ids"]]],
            dtype=torch.long,
        )
        mask = torch.tensor([inputs["attention_mask"]], dtype=torch.long)
        sent_len = sum(inputs["attention_mask"])
        tokens = TOKENIZER.convert_ids_to_tokens(inputs["input_ids"][1:sent_len-1])
        return input_ids, ref_input_ids, mask, sent_len, tokens 


    attributes = list()
    for sentence in request_json["sentences"]:
        input_ids, ref_input_ids, mask, sent_len, tokens = tokenize(sentence)

        attributions, delta = lig_attribute(
            forward_func, MODEL.distilbert.embeddings, input_ids, ref_input_ids, mask)

        attributes.append({
            "attributions": attributions[1:sent_len-1].tolist(),
            "delta": delta[0],
            "tokens": tokens,
        })

    return {"attributes": attributes}


# pylint: disable=invalid-name
app = Flask(__name__)


@app.route("/", methods=["POST"])
def get_prob():
    """Returns probability."""
    y_prob = predict(request.json)
    output = {"y_prob": y_prob}

    if request.json["bool_xai"] == 1:
        attributes = nlp_xai(request.json)
        output.update(attributes)
    return output


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()
