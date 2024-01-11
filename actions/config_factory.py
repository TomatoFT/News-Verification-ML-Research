from models.AlBERT import AlBERTNewsVerificationModel
from models.BERT import BERTNewsVerificationModel
from models.CamemBERT import CamemBERTNewsVerificationModel
from models.DistilBERT import DistilBERTNewsVerificationModel
from models.MobileBERT import MobileBERTNewsVerificationModel
from models.PhoBERT import PhoBERTNewsVerificationModel
from models.RetriBERT import RetriBERTNewsVerificationModel
from models.RoBERTa import RoBERTaNewsVerificationModel
from models.XLMNet import XLMNetNewsVerificationModel

from transformers import (AlbertTokenizer, AutoTokenizer, BertTokenizer,
                          CamembertTokenizer, DistilBertTokenizer,
                          MobileBertTokenizer, RobertaTokenizer,
                          XLMRobertaTokenizer)

class ModelTokenizerConfig:
    def __init__(self, tokenizer, name, model):
        self.tokenizer = tokenizer
        self.model = model
        self.name = name        

class ModelTokenizerFactory:
    def __init__(self, type) -> None:
        self.type = type
        self.model_dict = {
            'BERT': (BertTokenizer, 'bert-base-uncased', BERTNewsVerificationModel()),
            'DistilBERT': (DistilBertTokenizer, 'distilbert-base-uncased', DistilBERTNewsVerificationModel()),
            'RoBERTa': (RobertaTokenizer, 'roberta-base', RoBERTaNewsVerificationModel()),
            'AlBERT': (AlbertTokenizer, 'albert-base-v2', AlBERTNewsVerificationModel()),
            'RetriBERT': (BertTokenizer, 'yjernite/retribert-base-uncased', RetriBERTNewsVerificationModel()),
            'MobileBERT': (MobileBertTokenizer, 'google/mobilebert-uncased', MobileBERTNewsVerificationModel()),
            'XLMNet': (XLMRobertaTokenizer, 'xlm-roberta-base', XLMNetNewsVerificationModel()),
            'CamemBERT': (CamembertTokenizer, 'camembert-base', CamemBERTNewsVerificationModel()),
            'PhoBERT': (AutoTokenizer, 'vinai/phobert-base', PhoBERTNewsVerificationModel())
        }

    def get_model(self):
        if self.type not in self.model_dict:
            raise ValueError(f'The type is incorrect or the model {self.type} is not implemented yet!')
        
        _, _, model_instance = self.model_dict[self.type]
        return model_instance

    def get_tokenizer(self):
        if self.type not in self.model_dict:
            raise ValueError(f'The type is incorrect or the model {self.type} is not implemented yet!')
        
        tokenizer_cls, tokenizer_name, _ = self.model_dict[self.type]
        return tokenizer_cls.from_pretrained(tokenizer_name)

    def get_name(self):
        if self.type not in self.model_dict:
            raise ValueError(f'The type is incorrect or the model {self.type} is not implemented yet!')
        
        _, model_name, _ = self.model_dict[self.type]
        return model_name

