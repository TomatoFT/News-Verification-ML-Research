import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, texts, numeric_features, targets, tokenizer, max_length):
        self.texts = texts
        self.numeric_features = numeric_features
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        numeric = self.numeric_features[idx]
        target = self.targets[idx]

        encoded_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded_text['input_ids'].squeeze(0),
            'attention_mask': encoded_text['attention_mask'].squeeze(0),
            'numeric': torch.tensor(numeric, dtype=torch.float),
            'target': torch.tensor(target, dtype=torch.long)
        }
