import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BERTNewsVerificationModel(nn.Module):
    def __init__(self):
        super(BERTNewsVerificationModel, self).__init__()
        self._name = 'BERT'
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.conv_layer = nn.Conv1d(in_channels=self.bert.config.hidden_size, 
                                    out_channels=32, kernel_size=3, padding=1)
        self.fc_text = nn.Linear(32, 64)
        self.fc_numeric = nn.Linear(1, 64)
        self.fc_combined = nn.Linear(128, 2)
        nn.init.normal_(self.fc_combined.weight, std=0.02)
        nn.init.normal_(self.fc_combined.bias, 0)

    def forward(self, input_ids, attention_mask, numeric):
        _, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        output = output.unsqueeze(2)
        conv_output = self.conv_layer(output)
        conv_output = F.relu(conv_output)

        pooled_output = F.max_pool1d(conv_output, conv_output.size(2)).squeeze(2)

        text_features = self.fc_text(self.drop(pooled_output))
        numeric_features = self.fc_numeric(numeric.unsqueeze(1))
        combined = torch.cat((text_features, numeric_features), dim=1)
        combined = self.drop(combined)
        output = self.fc_combined(combined)
        return output