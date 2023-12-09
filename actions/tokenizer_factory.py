from transformers import BertTokenizer, DistilBertTokenizer


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
        else:
            raise ValueError(f'Tokenizer {self.type} not found!')