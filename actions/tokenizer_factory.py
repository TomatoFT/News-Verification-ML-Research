from transformers import (BertTokenizer, 
                          DistilBertTokenizer, 
                          RobertaTokenizer)


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
        else:
            raise ValueError(f'Tokenizer {self.type} not found!')