from models.BERT import BERTNewsVerificationModel
from models.DistilBERT import DistilBERTNewsVerificationModel
from models.RoBERTa import RoBERTaNewsVerificationModel
from models.AlBERT import AlBERTNewsVerificationModel


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
        elif self.type == 'AlBERT':
            return AlBERTNewsVerificationModel()
        else:
            raise ValueError(f'The type is incorrect. Model {self.type} is implemented yet!')