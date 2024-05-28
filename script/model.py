import math

import torch
from torch import nn


# from torch.nn import TransformerEncoderLayer, TransformerEncoder
# from torch_cluster import knn_graph, radius_graph
# from torch_geometric.nn import MetaLayer, voxel_grid, global_max_pool, max_pool, avg_pool
# import torch.nn.functional as F
# from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, to_dense_batch, to_dense_adj
# from torch_scatter import scatter_add, scatter_mean, scatter_max


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('encoding', self._get_timing_signal(max_len, d_model))

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

    def _get_timing_signal(self, length, channels):
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2) * -(math.log(10000.0) / channels))
        pe = torch.zeros(length, channels)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe


class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, max_len=5000):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=dim_feedforward)

    def forward(self, src, tgt):
        src = src + self.pos_encoder(src)
        # src = self.pos_encoder(src)
        output = self.transformer(src, tgt)
        return output


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=dilation * (kernel_size - 1) // 2),
                       nn.ReLU(),
                       nn.Dropout(dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = self.network(x.permute(0, 2, 1))
        x = self.linear(x.permute(0, 2, 1))    # Reshape for linear layer
        return x




class AIMP(torch.nn.Module):
    def __init__(self, pre_feas_dim, feas_dim, hidden, n_transformer, dropout):
        super(AIMP, self).__init__()

        self.pre_embedding = nn.Sequential(
            nn.Conv1d(pre_feas_dim, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

        self.embedding = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

        self.bn = nn.ModuleList([nn.BatchNorm1d(pre_feas_dim),
                                 nn.BatchNorm1d(feas_dim),
                                 # nn.BatchNorm1d(seq_feas_dim),
                                 ])

        # self.n_bilstm = n_bilstm
        self.n_transformer = n_transformer
        # self.lstm = nn.LSTM(input_size=hidden, hidden_size=hidden, num_layers=self.n_bilstm, bidirectional=True,
        #                     dropout=dropout, batch_first=True)
        #
        # self.lstm_act = nn.Sequential(
        #     nn.BatchNorm1d(hidden * 2),
        #     nn.ELU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Conv1d(hidden * 2, hidden, kernel_size=1),
        # )


        self.tcn = TCN(feas_dim, hidden, [hidden // 2, hidden // 2, hidden // 2], 21, dropout)
        self.tcn_res = nn.Sequential(
            nn.Conv1d(hidden + feas_dim, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

        # self.transformer = nn.Transformer(d_model=hidden, nhead=2, num_encoder_layers=self.n_transformer, batch_first=True)
        self.transformer = TransformerModel(d_model=hidden, nhead=4, num_layers=self.n_transformer,
                                            dim_feedforward=2048)
        self.transformer_act = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )
        self.transformer_res = nn.Sequential(
            nn.Conv1d(hidden + hidden, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )
        self.transformer_pool = nn.AdaptiveAvgPool2d((1, None))
        self.clf = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.pre_embedding, self.embedding, self.clf]:
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.zeros_(layer.bias)
        for layer in [self.transformer_act, self.transformer_res]:
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.zeros_(layer.bias)

    def forward(self, pre_feas, feas):
        # batch_size = pre_feas.shape[0]
        pre_feas = self.bn[0](pre_feas.permute(0, 2, 1)).permute(0, 2, 1)
        feas = self.bn[1](feas.permute(0, 2, 1)).permute(0, 2, 1)

        pre_feas = self.pre_embedding(pre_feas.permute(0, 2, 1)).permute(0, 2, 1)

        tcn_out = self.tcn(feas)
        tcn_out = self.tcn_res(torch.cat([tcn_out, feas], dim=-1).permute(0, 2, 1)).permute(0, 2, 1)

        feas_em = self.embedding(torch.cat([pre_feas, tcn_out], dim=-1).permute(0, 2, 1)).permute(0, 2, 1)

        transformer_out = self.transformer(feas_em, feas_em)
        transformer_out = self.transformer_act(transformer_out.permute(0, 2, 1)).permute(0, 2, 1)
        transformer_out = self.transformer_res(torch.cat([transformer_out, feas_em], dim=-1).permute(0, 2, 1)).permute(
            0, 2, 1)
        transformer_out = self.transformer_pool(transformer_out).squeeze(1)

        out = self.clf(transformer_out)
        out = torch.nn.functional.softmax(out, -1)
        return out[:, -1]
