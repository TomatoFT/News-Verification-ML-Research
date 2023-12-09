import torch
import torch.nn as nn
import yaml

from actions.get_dataloader import GetDataLoader
from actions.training import TrainingDeepLearningModel
from models.BERT import BERTNewsVerificationModel


def read_config(file_path: str):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = read_config(file_path='models/config.yaml')

dataloader = GetDataLoader(tokenizer_name='bert-base-uncased').get_dataloader(batch_size=2)
model = BERTNewsVerificationModel() 
optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['hyperparameters']['learning_rate']))
criterion = nn.CrossEntropyLoss()


training_deeplearning = TrainingDeepLearningModel(model=model, 
                          optimizer=optimizer, 
                          criterion=criterion, 
                          dataloader=dataloader, 
                          num_epochs=config['hyperparameters']['num_epochs'])

training_deeplearning.training(dry_run=True)

print('Done Training')