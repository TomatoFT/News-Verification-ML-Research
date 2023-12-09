import time

import torch


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
            

    def wet_run_training(self):
        self.model.to(self.device)

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for batch in self.dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                numeric = batch['numeric'].to(self.device)
                target = batch['target'].to(self.device)

                self.optimizer.zero_grad()

                output = self.model(input_ids, attention_mask, numeric)
                loss = self.criterion(output, target)
                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {avg_loss:.4f}")

        # Optionally, save the trained model
        torch.save(self.model.state_dict(), f'saved_models/{self.model._name}_news_verification.pth')

    def training(self, dry_run=True):
        if dry_run:
            self.dry_run_training()
        else:
            self.wet_run_training()
