# from model.sa_convLSTM_cell import SA_Convlstm_cell
from model.tls_convLSTM_cell import TLS_Convlstm_cell, FrontTensorProduct

import torch
import torch.nn as nn
import random

###
#
# Author: Min Namgung
# Contact: namgu007@umn.edu
#
# ###

class Encode2Decode(nn.Module):  # self-attention convlstm for spatiotemporal prediction model
    def __init__(self, config):
        super(Encode2Decode, self).__init__()
        # hyperparams
        self.device = torch.device(str(config['model']['device']))
        self.batch_size = int(config['model']['batch_size'])
        self.img_size = (int(config['data']['img_height']), int(config['data']['img_width']))
        self.cells, self.bns, self.decoderCells = [], [], []
        self.n_layers = int(config['model']['n_layers'])
        self.input_dim = int(config['model']['in_channel'])
        self.input_window_size = int(config['model']['input_len'])
        self.output_window_size = int(config['model']['output_len'])
        self.node_embed_dim = int(config['model']['node_embed_dim'])
        self.hidden_dim = int(config['model']['hidden_dim'])
        self.total_window_size = self.input_window_size + self.output_window_size
        self.config = config
        height = self.img_size[0]
        width = self.img_size[1]
        n_node = height * width
        n_relation = self.input_dim
        self.tproduct = FrontTensorProduct(height=height, width=width, num_relation=n_relation)
        
        # tensor node embeddings
        self.TensorNodeEmbeddings = nn.Parameter(torch.zeros(n_relation, n_node, self.node_embed_dim))
        nn.init.xavier_normal_(self.TensorNodeEmbeddings)
        # print("self.TensorNodeEmbeddings:",self.TensorNodeEmbeddings)
        
        self.identity = torch.zeros(n_relation, n_node, n_node).to(self.device)
        
        for n in range(n_node):
            self.identity[0][n][n] = 1
        

        # Written By Min
        self.img_encode = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim, kernel_size=1, stride=1, padding=0,
                      out_channels=self.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=self.hidden_dim, kernel_size=3, stride=2, padding=1,
                      out_channels=self.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=self.hidden_dim, kernel_size=3, stride=2, padding=1,
                      out_channels=self.hidden_dim),
            nn.LeakyReLU(0.1)
        )

        self.img_decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1,
                               out_channels=self.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(in_channels=self.hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1,
                               out_channels=self.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=self.hidden_dim, kernel_size=1, stride=1, padding=0,
                      out_channels=self.input_dim)
        )


        for i in range(self.n_layers):
            self.input_dim == self.hidden_dim if i == 0 else self.hidden_dim
            self.hidden_dim == self.hidden_dim
            self.cells.append(TLS_Convlstm_cell(self.config))
            self.bns.append(nn.LayerNorm((self.hidden_dim, self.img_size[0], self.img_size[1])))  # Use layernorm

        

        self.cells = nn.ModuleList(self.cells)

        self.bns = nn.ModuleList(self.bns)
        self.decoderCells = nn.ModuleList(self.decoderCells)

        # Linear
        self.decoder_predict = nn.Conv2d(in_channels=self.hidden_dim,
                                         out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0))

    def forward(self, x, y, teacher_forcing_rate=0.5, hidden=None):
        if hidden == None:
            hidden = self.init_hidden(batch_size=self.batch_size, img_size=self.img_size)

        b, seq_len, x_c, h, w = x.size()
        _, horizon, y_c, h, w = y.size()

        predict_temp_de = []

        in_x = min(x_c, y_c)
        # lag_y = torch.cat([x[:, -1:, :in_x, :, :], y[:, :-1, :in_x, :, :]], dim=1)

        frames = torch.cat([x, y], dim=1)
        
         # calculate tensor adj
        relu_adj = torch.relu(self.tproduct(self.TensorNodeEmbeddings, 
                                            self.TensorNodeEmbeddings.permute(0, 2, 1).contiguous()))
        adj = torch.softmax(relu_adj, dim=2)
        adj = adj + self.identity


        for t in range(self.total_window_size):

            if t < self.input_window_size or random.random() < teacher_forcing_rate:
                x = frames[:, t, :, :, :]
            else:
                x = out

            x = self.img_encode(x)

            for i, cell in enumerate(self.cells):

                if i == 0:
                    out, hidden[i] = cell(x, hidden[i], adj)
                    out = self.bns[i](out)

                else:
                    out, hidden[i] = cell(x, hidden[i], adj)
                    out = self.bns[i](out)

            # out = self.decoder_predict(out)
            out = self.img_decode(out)
            predict_temp_de.append(out)

        predict_temp_de = torch.stack(predict_temp_de, dim=1)

        predict_temp_de = predict_temp_de[:, self.input_window_size:, :, 3:, 3:]

        return predict_temp_de


    def init_hidden(self, batch_size, img_size):
        states = []
        for i in range(self.n_layers):
            states.append(self.cells[i].init_hidden(batch_size, img_size))

        return states