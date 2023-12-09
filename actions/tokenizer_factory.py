from transformers import (AlbertTokenizer, BertTokenizer, DistilBertTokenizer,
                          FlaubertTokenizer, MobileBertTokenizer,
                          RobertaTokenizer, XLMRobertaTokenizer,
                          CamembertTokenizer)


class Tokenizer:
    def __init__(self, tokenizer, name):
        self.tokenizer = tokenizer
        self.name = name


class TokenizerFactory:
    def __init__(self, type) -> None:
        self.type = type
            
    def get_tokenizer(self):
        if self.type == 'BERT':
            return Tokenizer(tokenizer=BertTokenizer, name='bert-base-uncased')
        elif self.type == 'DistilBERT':
            return Tokenizer(tokenizer=DistilBertTokenizer, name='distilbert-base-uncased')
        elif self.type == 'RoBERTa':
            return Tokenizer(tokenizer=RobertaTokenizer, name='roberta-base')
        elif self.type == 'AlBERT':
            return Tokenizer(tokenizer=AlbertTokenizer, name='albert-base-v2')
        elif self.type == 'FlauBERT':
            return Tokenizer(tokenizer=FlaubertTokenizer, name='flaubert/flaubert_base_cased')
        elif self.type == 'MobileBERT':
            return Tokenizer(tokenizer=MobileBertTokenizer, name='google/mobilebert-uncased')
        elif self.type == 'XLMNet':
            return Tokenizer(tokenizer=XLMRobertaTokenizer, name='xlm-roberta-base')
        elif self.type == 'CamemBERT':
            return Tokenizer(tokenizer=CamembertTokenizer, name='camembert-base')
        else:
            raise ValueError(f'Tokenizer {self.type} not found!')