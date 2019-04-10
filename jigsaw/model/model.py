import torch
import torch.nn as nn
import torch.nn.functional as F

from jigsaw.base.base_model import BaseModel


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)                          # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)                   # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)                   # (N, T, 1, K)
        x = x.squeeze(2)                            # (N, T, K)
        return x


class JigsawLstmModel(BaseModel):
    def __init__(self, embedding_matrix, num_aux_targets, max_features=None,
                 lstm_units=128, dense_hidden_units=4 * 128):
        super(JigsawLstmModel, self).__init__()
        embed_size = embedding_matrix.shape[1]
        max_features = max_features if max_features else embed_size

        print(f'Using embed_size={embed_size}, max_features={max_features}, '
              f'dense_hidden_units={dense_hidden_units}')

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)

        self.lstm1 = nn.LSTM(embed_size, lstm_units, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(dense_hidden_units, dense_hidden_units)
        self.linear2 = nn.Linear(dense_hidden_units, dense_hidden_units)

        self.linear_out = nn.Linear(dense_hidden_units, 1)
        self.linear_aux_out = nn.Linear(dense_hidden_units, num_aux_targets)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)

        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)

        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        h_conc_linear2  = F.relu(self.linear2(h_conc))

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)

        return out
