import torch
import torch.nn as nn
import yaml

from actions.evaluation import Evaluate
from actions.get_dataloader import GetDataLoader
from actions.training import TrainingDeepLearningModel
from models.BERT import BERTNewsVerificationModel


def read_config(file_path: str):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = read_config(file_path='models/config.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataloader = GetDataLoader(tokenizer_name='bert-base-uncased').get_dataloader(batch_size=2)
model = BERTNewsVerificationModel() 
optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['hyperparameters']['learning_rate']))
criterion = nn.CrossEntropyLoss()


training_deeplearning = TrainingDeepLearningModel(model=model, 
                          optimizer=optimizer, 
                          criterion=criterion, 
                          dataloader=dataloader, 
                          num_epochs=config['hyperparameters']['num_epochs'],
                          device=device)

training_deeplearning.training(dry_run=False)

print('Done Training')

evaluator = Evaluate(model, dataloader, device)
accuracy, precision, recall, f1 = evaluator.evaluate_model()

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

print('Done Evaluation')