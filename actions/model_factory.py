from models.BERT import BERTNewsVerificationModel
from models.DistilBERT import DistilBERTNewsVerificationModel


class ModelsFactory:
    def __init__(self, type) -> None:
        self.type = type
            
    def get_model(self):
        if self.type == 'BERT':
            return BERTNewsVerificationModel()
        elif self.type == 'DistilBERT':
            return DistilBERTNewsVerificationModel()
        else:
            raise ValueError(f'Model {self.type} not found!')