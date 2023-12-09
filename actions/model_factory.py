from models.AlBERT import AlBERTNewsVerificationModel
from models.BERT import BERTNewsVerificationModel
from models.CamemBERT import CamemBERTNewsVerificationModel
from models.DistilBERT import DistilBERTNewsVerificationModel
from models.FlauBERT import FlauBERTNewsVerificationModel
from models.MobileBERT import MobileBERTNewsVerificationModel
from models.PhoBERT import PhoBERTNewsVerificationModel
from models.RoBERTa import RoBERTaNewsVerificationModel
from models.XLMNet import XLMNetNewsVerificationModel


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
        elif self.type == 'FlauBERT':
            return FlauBERTNewsVerificationModel()
        elif self.type == 'MobileBERT':
            return MobileBERTNewsVerificationModel()
        elif self.type == 'XLMNet':
            return XLMNetNewsVerificationModel()
        elif self.type == 'CamemBERT':
            return CamemBERTNewsVerificationModel()
        elif self.type == 'PhoBERT':
            return PhoBERTNewsVerificationModel()
        else:
            raise ValueError(f'The type is incorrect. Model {self.type} is implemented yet!')