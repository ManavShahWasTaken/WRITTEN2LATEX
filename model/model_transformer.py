
from multiprocessing.dummy import active_children
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
from torch.distributions.uniform import Uniform
from torch import Tensor
import math

INIT = 1e-2

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

config=Config(
    feat_size=512,
    encoder_nheads=8,
    encoder_dim_feedfoward=2048,
    encoder_num_layers=6,
    encoder_dropout=0.2,
    encoder_max_sequence_length=1024,
    decoder_nheads=8,
    decoder_dim_feedfoward=2048,
    decoder_num_layers=8,
    decoder_dropout=0.2,
    decoder_max_sequence_length=1024
)

class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=config.encoding_dropout)

        position = torch.arange(config.encoder_max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.feat_size, 2) * (-math.log(10000.0) / config.feat_size))
        pe = torch.zeros(config.encoder_max_sequence_length, 1, config.feat_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class Text_to_embedding(nn.Module):
    def __init__(self, config):
        super(Text_to_embedding, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, embedding_dim=config.feat_size)
        self.embedding_size = config.feat_size
    
    def forward(self, token_idx: Tensor):
        return self.embedding(token_idx.long()) * math.sqrt(self.embedding_size)


class LatexTransformer(nn.Module):
    def __init__(self, config):
        super(LatexTransformer, self).__init__()
        self.model_type = 'Transformer'
        encoder_layer = TransformerEncoderLayer(d_model=config.feat_size, nhead=config.encoder_nheads,
                                                dim_feedforward=config.encoder_dim_feedforward, dropout=config.encoder_dropout, batch_first=True,
                                                activation="gelu")
        
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=config.encoder_num_layers)

        decoder_layer = TransformerDecoderLayer(d_model=config.feat_size, nhead=config.decoder_nheads,
                                                dim_feedforward=config.decoder_dim_feedforward, dropout=config.decoder_dropout,
                                                batch_first=True, activation="gelu")
        
        self.transformer_decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=config.decoder_num_layers)

        self.positional_encoding = PositionalEncoding(config)
        self.vocab_to_embedding = Text_to_embedding(config)
        self.embedding_to_vocab = nn.Linear(config.feat_size, config.vocab_size)
    
    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        
        src_emb = self.positional_encoding(src)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)
    
    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(src), src_mask)


class Im2LatexModel(nn.Module):
    def __init__(self, config):
        super(Im2LatexModel, self).__init__()

        self.config = config
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

            nn.Conv2d(256, config.feat_size, 3, 1, 0),
            nn.ReLU()
        )

        self.transformer = LatexTransformer(config)

    def encode(self, image_batch: Tensor):
        cnn_encoded = self.cnn_encoder(image_batch) # batch, feat_size, H, W, 

    
    def forward(self, x):
        pass




        # """
        # c:512, k:(3,3), s:(1,1), p:(0,0), bn -
        # c:512, k:(3,3), s:(1,1), p:(1,1), bn po:(1,2), s:(1,2), p:(0,0)
        # c:256, k:(3,3), s:(1,1), p:(1,1) po:(2,1), s:(2,1), p(0,0)
        # c:256, k:(3,3), s:(1,1), p:(1,1), bn -
        # c:128, k:(3,3), s:(1,1), p:(1,1) po:(2,2), s:(2,2), p:(0,0)
        # c:64, k:(3,3), s:(1,1), p:(1,1) po:(2,2), s:(2,2), p(2,2)
        # """

        # """
        # self.cnn_encoder = nn.Sequential(
        #     nn.Conv2d(1, 512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),

        #     nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 0)),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),

        #     nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.MaxPool2d((2, 1), stride=(2, 1), padding=(0, 0)),
        #     nn.ReLU(),

        #     nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),

        #     nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.MaxPool2d((2, 2), stride=(2, 2), padding=(0, 0)),
        #     nn.ReLU(),

        #     nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.MaxPool2d((2, 2), stride=(2, 2), padding=(0, 0)),
        #     nn.ReLU(),
        # )
        # """
        # # encoder and decoder
        # """
        #     self.row_encoder = nn.LSTM(enc_out_dim, enc_rnn_h, bidirectional=True, batch_first=True)
        #     self.rnn_decoder = nn.LSTMCell(dec_rnn_h+emb_size, dec_rnn_h)
        #     self.embedding = nn.Embedding(out_size, emb_size)

        #     self.init_wh = nn.Linear(enc_out_dim, dec_rnn_h)
        #     self.init_wc = nn.Linear(enc_out_dim, dec_rnn_h)
        #     self.init_wo = nn.Linear(enc_out_dim, dec_rnn_h)

        #     # Attention mechanism
        #     self.beta = nn.Parameter(torch.Tensor(enc_out_dim))
        #     init.uniform_(self.beta, -INIT, INIT)
        #     self.W_1 = nn.Linear(enc_out_dim, enc_out_dim, bias=False)
        #     self.W_2 = nn.Linear(dec_rnn_h, enc_out_dim, bias=False)

        #     self.W_3 = nn.Linear(dec_rnn_h+enc_out_dim, dec_rnn_h, bias=False)
        #     self.W_out = nn.Linear(dec_rnn_h, out_size, bias=False)

        #     self.add_pos_feat = add_pos_feat
        #     self.dropout = nn.Dropout(p=dropout)
        #     self.uniform = Uniform(0, 1)
        # """
        

    # def forward(self, imgs, formulas, epsilon=1.):
    #     """args:
    #     imgs: [B, C, H, W]
    #     formulas: [B, MAX_LEN]
    #     epsilon: probability of the current time step to
    #              use the true previous token
    #     return:
    #     logits: [B, MAX_LEN, VOCAB_SIZE]
    #     """
    #     # encoding
    #     encoded_imgs = self.encode(imgs)  # [B, H*W, 512]
    #     # init decoder's states
    #     dec_states, o_t = self.init_decoder(encoded_imgs)
    #     max_len = formulas.size(1)
    #     logits = []
    #     for t in range(max_len):
    #         tgt = formulas[:, t:t+1]
    #         # schedule sampling
    #         if logits and self.uniform.sample().item() > epsilon:
    #             tgt = torch.argmax(torch.log(logits[-1]), dim=1, keepdim=True)
    #         # ont step decoding
    #         dec_states, O_t, logit = self.step_decoding(
    #             dec_states, o_t, encoded_imgs, tgt)
    #         logits.append(logit)
    #     logits = torch.stack(logits, dim=1)  # [B, MAX_LEN, out_size]
    #     return logits

    # def encode(self, imgs):
    #     encoded_imgs = self.cnn_encoder(imgs)  # [B, 512, H', W']
    #     encoded_imgs = encoded_imgs.permute(0, 2, 3, 1)  # [B, H', W', 512]

    #     #print("encoded_imgs.shape", encoded_imgs.shape)

    #     # ADD ROW ENCODER HERE
    #     H = encoded_imgs.shape[1]
    #     outputs = []
    #     for h in range(H):
    #         row_features = encoded_imgs[:, h, :, :] # [B, W', 512] - (batch, seq, feature)
    #         #print("row_features.shape", row_features.shape)
    #         output, h_n = self.row_encoder(row_features)

    #         #print("output.shape", output.shape)

    #         outputs.append(output)

    #     row_encoded = torch.stack(outputs, dim=1)

    #     # Row encoder positional embeddings?
    #     # Initial hidden states?

    #     B, H, W, _ = row_encoded.shape
    #     row_encoded = row_encoded.contiguous().view(B, H*W, -1)
    #     if self.add_pos_feat:
    #         row_encoded = add_positional_features(row_encoded)

    #     return row_encoded

    #     #return encoded_imgs

    # def step_decoding(self, dec_states, o_t, enc_out, tgt):
    #     """Runing one step decoding"""

    #     prev_y = self.embedding(tgt).squeeze(1)  # [B, emb_size]
    #     inp = torch.cat([prev_y, o_t], dim=1)  # [B, emb_size+dec_rnn_h]
    #     h_t, c_t = self.rnn_decoder(inp, dec_states)  # h_t:[B, dec_rnn_h]
    #     h_t = self.dropout(h_t)
    #     c_t = self.dropout(c_t)

    #     # context_t : [B, C]
    #     context_t, attn_scores = self._get_attn(enc_out, h_t)

    #     # [B, dec_rnn_h]
    #     o_t = self.W_3(torch.cat([h_t, context_t], dim=1)).tanh()
    #     o_t = self.dropout(o_t)

    #     # calculate logit
    #     logit = F.softmax(self.W_out(o_t), dim=1)  # [B, out_size]

    #     return (h_t, c_t), o_t, logit

    # def _get_attn(self, enc_out, h_t):
    #     """Attention mechanism
    #     args:
    #         enc_out: row encoder's output [B, L=H*W, C]
    #         h_t: the current time step hidden state [B, dec_rnn_h]
    #     return:
    #         context: this time step context [B, C]
    #         attn_scores: Attention scores
    #     """
    #     # cal alpha
    #     alpha = torch.tanh(self.W_1(enc_out)+self.W_2(h_t).unsqueeze(1))
    #     alpha = torch.sum(self.beta*alpha, dim=-1)  # [B, L]
    #     alpha = F.softmax(alpha, dim=-1)  # [B, L]

    #     # cal context: [B, C]
    #     context = torch.bmm(alpha.unsqueeze(1), enc_out)
    #     context = context.squeeze(1)
    #     return context, alpha

    # def init_decoder(self, enc_out):
    #     """args:
    #         enc_out: the output of row encoder [B, H*W, C]
    #       return:
    #         h_0, c_0:  h_0 and c_0's shape: [B, dec_rnn_h]
    #         init_O : the average of enc_out  [B, dec_rnn_h]
    #         for decoder
    #     """
    #     mean_enc_out = enc_out.mean(dim=1)
    #     h = self._init_h(mean_enc_out)
    #     c = self._init_c(mean_enc_out)
    #     init_o = self._init_o(mean_enc_out)
    #     return (h, c), init_o

    # def _init_h(self, mean_enc_out):
    #     return torch.tanh(self.init_wh(mean_enc_out))

    # def _init_c(self, mean_enc_out):
    #     return torch.tanh(self.init_wc(mean_enc_out))

    # def _init_o(self, mean_enc_out):
    #     return torch.tanh(self.init_wo(mean_enc_out))
