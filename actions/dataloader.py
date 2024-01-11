import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.titles = [d['title'] for d in data]
        self.contents = [d['content'] for d in data]
        self.numeric_features = [[d['credit'], d['media']] for d in data]  # Added numeric features
        self.targets = [d['target'] for d in data]

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):
        title = str(self.titles[idx])
        content = str(self.contents[idx])
        numeric = torch.tensor(self.numeric_features[idx], dtype=torch.float)
        target = torch.tensor(self.targets[idx], dtype=torch.long)

        # Concatenate title and content
        concatenated_text = title + " " + content

        encoded_text = self.tokenizer.encode_plus(
            concatenated_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded_text['input_ids'].squeeze(0),
            'attention_mask': encoded_text['attention_mask'].squeeze(0),
            'numeric': numeric,
            'target': target
        }