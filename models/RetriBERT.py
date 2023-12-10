import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class RetriBERTNewsVerificationModel(nn.Module):
    def __init__(self):
        super(RetriBERTNewsVerificationModel, self).__init__()
        self.bert_model = BertModel.from_pretrained('yjernite/retribert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.fc_numeric = nn.Linear(1, 64)
        self.fc_combined = nn.Linear(768 + 64, 2)
        nn.init.normal_(self.fc_combined.weight, std=0.02)
        nn.init.normal_(self.fc_combined.bias, 0)

    def forward(self, input_ids, attention_mask, numeric):
        bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled_output = bert_output.pooler_output

        numeric_features = self.fc_numeric(numeric.unsqueeze(1))
        combined = torch.cat((pooled_output, numeric_features), dim=1)
        combined = self.drop(combined)
        output = self.fc_combined(combined)
        return output
