import torch


class Evaluate:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def accuracy(self, total_correct, total_samples):
        return total_correct / total_samples

    def precision_recall_f1(self, predicted, target):
        # Calculate true positive, true negative, false positive, false negative
        tp = torch.sum((predicted == 1) & (target == 1)).item()
        tn = torch.sum((predicted == 0) & (target == 0)).item()
        fp = torch.sum((predicted == 1) & (target == 0)).item()
        fn = torch.sum((predicted == 0) & (target == 1)).item()

        # Calculate precision, recall, and f1 score
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * (precision * recall) / max(precision + recall, 1)

        return precision, recall, f1

    def evaluate_model(self):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        all_predicted = []
        all_target = []

        with torch.no_grad():
            for batch in self.dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                numeric = batch['numeric'].to(self.device)
                target = batch['target'].to(self.device)

                output = self.model(input_ids, attention_mask, numeric)
                _, predicted = torch.max(output, dim=1)

                all_predicted.extend(predicted.cpu().numpy())
                all_target.extend(target.cpu().numpy())

                total_samples += target.size(0)
                total_correct += (predicted == target).sum().item()

        accuracy_value = self.accuracy(total_correct, total_samples)
        precision, recall, f1 = self.precision_recall_f1(torch.tensor(all_predicted), torch.tensor(all_target))

        return accuracy_value, precision, recall, f1


