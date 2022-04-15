import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.distributions.uniform import Uniform

import code # DEBUGGING

from .position_embedding import add_positional_features

INIT = 1e-2


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe



class Im2LatexModel(nn.Module):
    def __init__(self, experiment_dict, out_size, emb_size, \
        enc_rnn_h, dec_rnn_h, device, \
        enc_out_dim=512,  n_layer=1, \
        dropout=0.):

        super(Im2LatexModel, self).__init__()
        
        self.experiment_dict = experiment_dict

        self.device = device

        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 1),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 1),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1), 0),

            nn.Conv2d(256, enc_out_dim, 3, 1, 0),
            nn.ReLU()
        )

        """
        c:512, k:(3,3), s:(1,1), p:(0,0), bn -
        c:512, k:(3,3), s:(1,1), p:(1,1), bn po:(1,2), s:(1,2), p:(0,0)
        c:256, k:(3,3), s:(1,1), p:(1,1) po:(2,1), s:(2,1), p(0,0)
        c:256, k:(3,3), s:(1,1), p:(1,1), bn -
        c:128, k:(3,3), s:(1,1), p:(1,1) po:(2,2), s:(2,2), p:(0,0)
        c:64, k:(3,3), s:(1,1), p:(1,1) po:(2,2), s:(2,2), p(2,2)
        """

        """
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=(0, 0)),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d((2, 2), stride=(2, 2), padding=(0, 0)),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d((2, 2), stride=(2, 2), padding=(0, 0)),
            nn.ReLU(),
        )
        """

        self.enc_rnn_h = enc_rnn_h
        self.row_encoder = nn.LSTM(enc_out_dim, enc_rnn_h, bidirectional=True, batch_first=True)

        self.rnn_decoder = nn.LSTMCell(dec_rnn_h+emb_size, dec_rnn_h)
        self.embedding = nn.Embedding(out_size, emb_size)

        self.init_wh = nn.Linear(enc_out_dim, dec_rnn_h)
        self.init_wc = nn.Linear(enc_out_dim, dec_rnn_h)
        self.init_wo = nn.Linear(enc_out_dim, dec_rnn_h)

        # Attention mechanism
        self.beta = nn.Parameter(torch.Tensor(enc_out_dim))
        init.uniform_(self.beta, -INIT, INIT)
        self.W_1 = nn.Linear(enc_out_dim, enc_out_dim, bias=False)
        self.W_2 = nn.Linear(dec_rnn_h, enc_out_dim, bias=False)

        self.W_3 = nn.Linear(dec_rnn_h+enc_out_dim, dec_rnn_h, bias=False)
        self.W_out = nn.Linear(dec_rnn_h, out_size, bias=False)

        self.dropout = nn.Dropout(p=dropout)
        self.uniform = Uniform(0, 1)

    def forward(self, imgs, formulas, epsilon=1.):
        """args:
        imgs: [B, C, H, W]
        formulas: [B, MAX_LEN]
        epsilon: probability of the current time step to
                 use the true previous token
        return:
        logits: [B, MAX_LEN, VOCAB_SIZE]
        """
        # encoding
        encoded_imgs = self.encode(imgs)  # [B, H*W, 512]
        #code.interact(local=locals())
        # init decoder's states
        dec_states, o_t = self.init_decoder(encoded_imgs)
        max_len = formulas.size(1)
        logits = []
        for t in range(max_len):
            tgt = formulas[:, t:t+1]
            # schedule sampling
            if logits and self.uniform.sample().item() > epsilon:
                tgt = torch.argmax(torch.log(logits[-1]), dim=1, keepdim=True)
            # ont step decoding
            dec_states, O_t, logit = self.step_decoding(
                dec_states, o_t, encoded_imgs, tgt)
            logits.append(logit)
        logits = torch.stack(logits, dim=1)  # [B, MAX_LEN, out_size]
        return logits

    def row_encoder_fn(self, encoded):
        H = encoded.shape[1]
        
        if self.experiment_dict["paper_emb"]:
            # Get positional embedding vectors for each row
            h0s = positionalencoding1d(self.enc_rnn_h, H)

        outputs = []
        for h in range(H):
            row_features = encoded[:, h, :, :] # [B, W', 512] - (batch, seq, feature)

            if self.experiment_dict["paper_emb"]:
                # First dim is 2 since bidirectional
                h0 = h0s[h].unsqueeze(0).repeat(2, 1).unsqueeze(1) # [2, 1, enc_rnn_h]
                c0 = torch.zeros(2, 1, self.enc_rnn_h) # [2, 1, enc_rnn_h]

                output, h_n = self.row_encoder(row_features,
                    (h0.to(self.device), c0.to(self.device)))
            else:
                output, h_n = self.row_encoder(row_features)

            outputs.append(output)

        encoded = torch.stack(outputs, dim=1) # [B, H', W', 512]
        
        return encoded
    
    def encode(self, imgs):
        encoded = self.cnn_encoder(imgs)  # [B, 512, H', W']
        encoded = encoded.permute(0, 2, 3, 1)  # [B, H', W', 512]
        
        # Apply row encoder (leads to the same shape as encoded)
        if self.experiment_dict["row_encoder"]:
            encoded = self.row_encoder_fn(encoded)

        B, H, W, _ = encoded.shape
        encoded = encoded.contiguous().view(B, H*W, -1)
        
        if self.experiment_dict["new_emb"]:
            encoded = add_positional_features(encoded)
            
        return encoded

    
    def step_decoding(self, dec_states, o_t, enc_out, tgt):
        """Runing one step decoding"""

        prev_y = self.embedding(tgt).squeeze(1)  # [B, emb_size]
        inp = torch.cat([prev_y, o_t], dim=1)  # [B, emb_size+dec_rnn_h]
        h_t, c_t = self.rnn_decoder(inp, dec_states)  # h_t:[B, dec_rnn_h]
        h_t = self.dropout(h_t)
        c_t = self.dropout(c_t)

        # context_t : [B, C]
        context_t, attn_scores = self._get_attn(enc_out, h_t)

        # [B, dec_rnn_h]
        o_t = self.W_3(torch.cat([h_t, context_t], dim=1)).tanh()
        o_t = self.dropout(o_t)

        # calculate logit
        logit = F.softmax(self.W_out(o_t), dim=1)  # [B, out_size]

        return (h_t, c_t), o_t, logit

    def _get_attn(self, enc_out, h_t):
        """Attention mechanism
        args:
            enc_out: row encoder's output [B, L=H*W, C]
            h_t: the current time step hidden state [B, dec_rnn_h]
        return:
            context: this time step context [B, C]
            attn_scores: Attention scores
        """
        # cal alpha
        alpha = torch.tanh(self.W_1(enc_out)+self.W_2(h_t).unsqueeze(1))
        alpha = torch.sum(self.beta*alpha, dim=-1)  # [B, L]
        alpha = F.softmax(alpha, dim=-1)  # [B, L]

        # cal context: [B, C]
        context = torch.bmm(alpha.unsqueeze(1), enc_out)
        context = context.squeeze(1)
        return context, alpha

    def init_decoder(self, enc_out):
        """args:
            enc_out: the output of row encoder [B, H*W, C]
          return:
            h_0, c_0:  h_0 and c_0's shape: [B, dec_rnn_h]
            init_O : the average of enc_out  [B, dec_rnn_h]
            for decoder
        """
        mean_enc_out = enc_out.mean(dim=1)
        h = self._init_h(mean_enc_out)
        c = self._init_c(mean_enc_out)
        init_o = self._init_o(mean_enc_out)
        return (h, c), init_o

    def _init_h(self, mean_enc_out):
        return torch.tanh(self.init_wh(mean_enc_out))

    def _init_c(self, mean_enc_out):
        return torch.tanh(self.init_wc(mean_enc_out))

    def _init_o(self, mean_enc_out):
        return torch.tanh(self.init_wo(mean_enc_out))
