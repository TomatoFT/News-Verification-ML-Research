import time

import torch
from tqdm import tqdm


class TrainingDeepLearningModel:
    def __init__(self, model, optimizer, criterion, dataloader, num_epochs, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.device = device

    def dry_run_training(self):
        for epoch in range (self.num_epochs):
            print(f"Training {self.model._name} in dry run mode. Epoch {epoch+1}")
            time.sleep(1)            

    def wet_run_training(self, is_saved=True, output_dir=None):
        self.model.to(self.device)

        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as t_batch:
            for batch in t_batch:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                numeric = batch['numeric'].to(device)
                target = batch['target'].to(device)

                optimizer.zero_grad()

                output = model(input_ids, attention_mask, numeric)
                loss = criterion(output, target)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

                t_batch.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
        
        if is_saved:
            self.save_model(output_dir=output_dir)

    def save_model(self, output_dir=None):
        if not output_dir:
            output_dir = f'saved_models/{self.model._name}_news_verification.pth'
        print('Model is saved in ', output_dir)
        return torch.save(self.model.state_dict(), output_dir)

    def training(self, dry_run=True, is_saved=False, output_dir=None):
        if dry_run:
            self.dry_run_training()
        else:
            self.wet_run_training(is_saved=is_saved, output_dir=output_dir)
