import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, labels=None):
        self.len = len(data)
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        sentence = self.data[index]
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
        )

        output = {
            "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
        }
        if self.labels is not None:
            output["targets"] = torch.tensor(self.labels[index], dtype=torch.long)
        return output
    
    def __len__(self):
        return self.len
