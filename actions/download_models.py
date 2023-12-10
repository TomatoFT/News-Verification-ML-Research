from transformers import (AlbertModel, AlbertTokenizer, AutoModel,
                          AutoTokenizer, BertModel, BertTokenizer,
                          CamembertModel, CamembertTokenizer, DistilBertModel,
                          DistilBertTokenizer, MobileBertModel,
                          MobileBertTokenizer, RobertaModel, RobertaTokenizer,
                          XLMRobertaModel, XLMRobertaTokenizer)


def download_all_models_and_tokenizers():
    BertModel.from_pretrained('bert-base-uncased')
    BertTokenizer.from_pretrained('bert-base-uncased')

    DistilBertModel.from_pretrained('distilbert-base-uncased')
    DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    RobertaModel.from_pretrained('roberta-base')
    RobertaTokenizer.from_pretrained('roberta-base')

    AlbertModel.from_pretrained('albert-base-v2')
    AlbertTokenizer.from_pretrained('albert-base-v2')

    BertModel.from_pretrained('yjernite/retribert-base-uncased')
    BertTokenizer.from_pretrained('yjernite/retribert-base-uncased')

    MobileBertModel.from_pretrained('google/mobilebert-uncased')
    MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')

    XLMRobertaModel.from_pretrained('xlm-roberta-base')
    XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

    CamembertModel.from_pretrained('camembert-base')
    CamembertTokenizer.from_pretrained('camembert-base')

    AutoModel.from_pretrained('vinai/phobert-base')
    AutoTokenizer.from_pretrained('vinai/phobert-base')

