from models.BERT import BERTNewsVerificationModel
from models.DistilBERT import DistilBERTNewsVerificationModel
from models.RoBERTa import RoBERTaNewsVerificationModel


class ModelsFactory:
    def __init__(self, type) -> None:
        self.type = type
            
    def get_model(self):
        if self.type == 'BERT':
            return BERTNewsVerificationModel()
        elif self.type == 'DistilBERT':
            return DistilBERTNewsVerificationModel()
        elif self.type == 'RoBERTa':
            return RoBERTaNewsVerificationModel()
        else:
            raise ValueError(f'Model {self.type} not found!')