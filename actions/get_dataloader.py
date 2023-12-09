from torch.utils.data import DataLoader

from actions.dataloader import CustomDataset


class GetDataLoader:
    def __init__(self, tokenizer_type, data=None):
        self.tokenizer_type = tokenizer_type
        if not data:
            self.data = [
                    {'text': 'Tôi thích PyTorch rất nhiều', 'numeric': 3.14, 'target': 1},
                    {'text': 'Deep learning là học sâu', 'numeric': 2.71, 'target': 0},
                    {'text': 'PyTorch là thư viện deep learning', 'numeric': 1.618, 'target': 1},
                    {'text': 'Machine learning rất là hay', 'numeric': 0.577, 'target': 0}
                    ]
        else:
            self.data = data

    def get_dataset(self):
        texts = [d['text'] for d in self.data]
        numeric_features = [d['numeric'] for d in self.data]
        targets = [d['target'] for d in self.data]

        # Initialize the tokenizer and prepare the dataset
        tokenizer = self.tokenizer_type.tokenizer.from_pretrained(self.tokenizer_type.name)
        max_length = 32
        dataset = CustomDataset(texts, numeric_features, targets, tokenizer, max_length)

        return dataset
    
    def get_dataloader(self, batch_size=2):
        dataset = self.get_dataset()
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)