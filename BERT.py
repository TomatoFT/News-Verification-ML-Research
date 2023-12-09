import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from models.BERT import BERTNewsVerificationModel
from preprocess.dataloader import CustomDataset
from preprocess.training import TrainingDeepLearningModel

# Example data (replace this with your dataset)
data = [
    {'text': 'Tôi thích PyTorch rất nhiều', 'numeric': 3.14, 'target': 1},
    {'text': 'Deep learning là học sâu', 'numeric': 2.71, 'target': 0},
    {'text': 'PyTorch là thư viện deep learning', 'numeric': 1.618, 'target': 1},
    {'text': 'Machine learning rất là hay', 'numeric': 0.577, 'target': 0}
]

texts = [d['text'] for d in data]
numeric_features = [d['numeric'] for d in data]
targets = [d['target'] for d in data]

# Initialize the tokenizer and prepare the dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 32

dataset = CustomDataset(texts, numeric_features, targets, tokenizer, max_length)

# Define hyperparameters and create DataLoader
batch_size = 2
learning_rate = 1e-5
num_epochs = 5


dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, optimizer, and loss function
model = BERTNewsVerificationModel()  # Assuming binary classification (2 classes)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


training_deeplearning = TrainingDeepLearningModel(model=model, 
                          optimizer=optimizer, 
                          criterion=criterion, 
                          dataloader=dataloader, 
                          num_epochs=num_epochs)

training_deeplearning.training(dry_run=False)

print('Done Training')