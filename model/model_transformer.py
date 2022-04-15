
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
from torch import Tensor
import math

INIT = 1e-2

# class Config:
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)

# config=Config(
#     feat_size=512,
#     encoder_nheads=8,
#     encoder_dim_feedforward=2048,
#     encoder_num_layers=6,
#     encoder_dropout=0.2,
#     encoder_max_sequence_length=1024,
#     decoder_nheads=8,
#     decoder_dim_feedforward=2048,
#     decoder_num_layers=8,
#     decoder_dropout=0.2,
#     decoder_max_sequence_length=1024
# )

class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=config.encoder_dropout)

        position = torch.arange(config.encoder_max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.feat_size, 2) * (-math.log(2000000.0) / config.feat_size))
        pe = torch.zeros(1, config.encoder_max_sequence_length, config.feat_size)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor, width: int=None) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if width is not None:
            assert x.shape[1]%width==0
            height = x.shape[1]//width
            row_pos_encoding = self.pe[:, :width, :].repeat(1, height, 1) # repeat encoding for each row
            row_pos_encoding = row_pos_encoding + torch.repeat_interleave(self.pe[:, :height, :], width, dim=1) # add encoding to differentiate each row
            x = x + (row_pos_encoding/2)
            return self.dropout(x)
        else:
            x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class Idx_to_embedding(nn.Module):
    def __init__(self, config):
        super(Idx_to_embedding, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, embedding_dim=config.feat_size)
        self.embedding_size = config.feat_size
    
    def forward(self, token_idx: Tensor):
        return self.embedding(token_idx.long()) * math.sqrt(self.embedding_size)


class LatexTransformer(nn.Module):
    def __init__(self, config):
        super(LatexTransformer, self).__init__()
        self.model_type = 'Transformer'

        self.tgt_to_emb = Idx_to_embedding(config)

        encoder_layer = TransformerEncoderLayer(d_model=config.feat_size, nhead=config.encoder_nheads, dim_feedforward=config.encoder_dim_feedforward,batch_first=True, dropout=config.encoder_dropout, activation="gelu")
        
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=config.encoder_num_layers)

        decoder_layer = TransformerDecoderLayer(d_model=config.feat_size, nhead=config.decoder_nheads,
                                                dim_feedforward=config.decoder_dim_feedforward, batch_first=True, dropout=config.decoder_dropout, activation="gelu")
        
        self.transformer_decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=config.decoder_num_layers)

        self.positional_encoding = PositionalEncoding(config)
        self.vocab_to_embedding = Idx_to_embedding(config)
        self.embedding_to_vocab = nn.Linear(config.feat_size, config.vocab_size)
    
    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor = None):
        
        memory = self.encode(src, src_padding_mask)     
        logits = self.decode(trg, memory, tgt_mask, tgt_padding_mask)
        return logits 
    
    def encode(self, src: Tensor, src_mask: Tensor=None, width=None):
        src_emb = self.positional_encoding(src, width=width)
        return self.transformer_encoder(src_emb, None, None)
    
    def decode(self, target: Tensor, memory: Tensor, target_mask: Tensor, target_padding_mask: Tensor):
        tgt_emb = self.positional_encoding(self.tgt_to_emb(target))
        outs = self.transformer_decoder(tgt_emb, memory, target_mask, None,
                                        target_padding_mask, None)
        return self.embedding_to_vocab(outs)


class Im2LatexModelTransformer(nn.Module):
    def __init__(self, config):
        super(Im2LatexModelTransformer, self).__init__()
        self.model_type = 'Im2LatexModelTransformer'
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

        
        self.use_patches = config.use_patches
        if config.use_patches:
            self.patch_size=config.patch_size
            self.patcher = nn.Conv2d(
                in_channels=config.feat_size,
                out_channels = config.feat_size,
                kernel_size=config.patch_size,
                stride=config.patch_size,
            )
        else:
            self.patcher = nn.Identity()  

        self.transformer = LatexTransformer(config)

    def encode(self, image_batch: Tensor):
        """
        input: Batch of images
        output: Batch x (k) x feat_size
        """
        temp = self.cnn_encoder(image_batch)
        change = False
        
        if self.use_patches:
            shape = list(temp.shape)
            if shape[2]%self.patch_size != 0:
                change=True
                shape[2] += shape[2]%self.patch_size
            if shape[3]%self.patch_size != 0:
                change=True
                shape[3] += shape[3]%self.patch_size
        
        if change:
            padding = torch.zeros(tuple(shape), device=image_batch.device)
            padding[:, :, :temp.shape[2], :temp.shape[3]] = temp
            temp = padding
        
        cnn_encoded = self.patcher(temp).permute(0, 2, 3, 1) # batch, H, W, feat_size

        width = cnn_encoded.shape[2]
        cnn_encoded = cnn_encoded.reshape(cnn_encoded.shape[0], cnn_encoded.shape[1] * cnn_encoded.shape[2], self.config.feat_size)
        transformed = self.transformer.encode(cnn_encoded, src_mask=None, width=width) # encode with positional embeddings
        return transformed
    
    def decode(self, target, memory, target_mask, target_padding_mask):
        return self.transformer.decode(target=target, \
            memory=memory,
            target_mask=target_mask,
            target_padding_mask=target_padding_mask)


    def forward(self, imgs, targets, target_mask, target_padding_mask):
        memory = self.encode(imgs) # attention vectors from enconding
        logits = self.decode(target=targets, memory=memory, target_mask=target_mask, target_padding_mask=target_padding_mask)
        return logits 
        
# if __name__ == '__main__':
#     config.vocab_size=353
#     model = Im2LatexModelTransformer(config)
#     batch_size = 8
#     seq_len = 40
#     h = 30
#     w = 400
#     c = 3
#     imgs = torch.ones(batch_size, c, h, w)
#     tgt = torch.ones(batch_size, seq_len)
#     mask = (torch.triu(torch.ones((seq_len, seq_len))) == 1).transpose(0, 1)
#     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#     tgt_padding = tgt == 0
#     x = model(imgs, tgt, mask, tgt_padding)
#     import code
#     code.interact(local=locals())
    