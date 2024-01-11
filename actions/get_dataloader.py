from torch.utils.data import DataLoader

from actions.dataloader import CustomDataset


class GetDataLoader:
    def __init__(self, tokenizer_type, data=None):
        self.tokenizer_type = tokenizer_type
        if not data:
            self.data = [
                        {'credit': 0.5, 'title': 'Title 1', 'content': 'Content 1', 'media': 3.14, 'target': 1},
                        {'credit': 0.5, 'title': 'Title 2', 'content': 'Content 2', 'media': 2.71, 'target': 0},
                        {'credit': 0.5, 'title': 'Title 3', 'content': 'Content 3', 'media': 1.618, 'target': 1},
                        {'credit': 0.5, 'title': 'Title 4', 'content': 'Content 4', 'media': 0.577, 'target': 0}
                    ]
        else:
            self.data = data

    def get_dataset(self):

        # Initialize the tokenizer and prepare the dataset
        tokenizer = self.tokenizer_type.tokenizer.from_pretrained(self.tokenizer_type.name)
        max_length = 32
        dataset = CustomDataset(self.data, tokenizer, max_length)

        return dataset
    
    def get_dataloader(self, batch_size=2):
        dataset = self.get_dataset()
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)