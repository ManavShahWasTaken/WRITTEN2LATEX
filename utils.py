import os
import math
from tkinter import image_names

import torch
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

from model.model_transformer import Im2LatexModelTransformer

from build_vocab import END_TOKEN, PAD_TOKEN, UNK_TOKEN, START_TOKEN


def collate_fn(sign2id, batch):
    # filter the pictures that have different width or height
    size = batch[0][0].size()
    batch = [img_formula for img_formula in batch
             if img_formula[0].size() == size]
    # sort by the length of formula
    batch.sort(key=lambda img_formula: len(img_formula[1].split()),
               reverse=True)
    imgs, formulas = zip(*batch)
    formulas = [formula.split() for formula in formulas]
    # targets for training , begin with START_TOKEN
    tgt4training = formulas2tensor(add_start_token(formulas), sign2id)
    # targets for calculating loss , end with END_TOKEN
    tgt4cal_loss = formulas2tensor(add_end_token(formulas), sign2id)
    imgs = torch.stack(imgs, dim=0)
    return imgs, tgt4training, tgt4cal_loss

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(tgt):
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    tgt_padding_mask = (tgt == PAD_TOKEN)
    assert tgt_padding_mask.shape[0] == tgt.shape[0] # batch dim
    assert tgt_padding_mask.shape[1] == tgt.shape[1] # target sequence dim

    return tgt_mask, tgt_padding_mask

def collate_transformer_fn(sign2id, batch):
    max_seq_length = 0
    h_max = 0
    w_max = 0
    formulas_train = []
    formulas_loss = []
    imgs = []

    for image, formula in batch:
        h_max = max(h_max, image.shape[1])
        w_max = max(w_max, image.shape[2])
        label = formula.split()
        formulas_train.append(['<s>'] + label)
        formulas_loss.append(label + ['</s>'])
        max_seq_length = max(max_seq_length, len(label)+1)
        imgs.append(image)
        
    targets_train = formulas2tensor(formulas_train, sign2id, max_len=max_seq_length)
    targets_loss = formulas2tensor(formulas_loss, sign2id, max_len=max_seq_length)
    images_tensor = []    
    for image in imgs:
        result = torch.zeros(3, h_max, w_max)
        result[:, :image.shape[1], :image.shape[2]] = image
        images_tensor.append(result)
    del imgs
    targets = (targets_train, targets_loss)
    # target_mask, target_padding_mask = create_mask(targets)
    image_tensor = torch.stack(images_tensor, dim=0)
    return image_tensor, targets # , target_padding_mask, target_mask

def formulas2tensor(formulas, sign2id, max_len=None):
    """convert formula to tensor"""
    batch_size = len(formulas)
    if max_len is None:
        max_len = len(formulas[0])
    
    tensors = torch.ones(batch_size, max_len, dtype=torch.long) * PAD_TOKEN
    for i, formula in enumerate(formulas):
        for j, sign in enumerate(formula):
            tensors[i][j] = sign2id.get(sign, UNK_TOKEN)
    return tensors


def add_start_token(formulas):
    return [['<s>']+formula for formula in formulas]


def add_end_token(formulas):
    return [formula+['</s>'] for formula in formulas]

# def add_start_stop_token(formulas):
#     return [+formula+['</s>'] for formula in formulas]


def count_parameters(model):
    """count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def load_formulas(filename):
    formulas = dict()
    with open(filename) as f:
        for idx, line in enumerate(f):
            formulas[idx] = line.strip()
    print("Loaded {} formulas from {}".format(len(formulas), filename))
    return formulas


def cal_loss(logits, targets):
    """args:
        logits: probability distribution return by model
                [B, MAX_LEN, voc_size]
        targets: target formulas
                [B, MAX_LEN]
    """
    padding = torch.ones_like(targets) * PAD_TOKEN
    mask = (targets != padding)

    targets = targets.masked_select(mask)
    logits = logits.masked_select(
        mask.unsqueeze(2).expand(-1, -1, logits.size(2))
    ).contiguous().view(-1, logits.size(2))
    logits = torch.log(logits)

    assert logits.size(0) == targets.size(0)

    loss = F.nll_loss(logits, targets)
    return loss


def get_checkpoint(ckpt_dir):
    """return full path if there is ckpt in ckpt_dir else None"""
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError("No checkpoint found in {}".format(ckpt_dir))

    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith('ckpt')]
    if not ckpts:
        raise FileNotFoundError("No checkpoint found in {}".format(ckpt_dir))

    last_ckpt, max_epoch = None, 0
    for ckpt in ckpts:
        epoch = int(ckpt.split('-')[1])
        if epoch > max_epoch:
            max_epoch = epoch
            last_ckpt = ckpt
    full_path = os.path.join(ckpt_dir, last_ckpt)
    print("Get checkpoint from {} for training".format(full_path))
    return full_path


def schedule_sample(prev_logit, prev_tgt, epsilon):
    prev_out = torch.argmax(prev_logit, dim=1, keepdim=True)
    prev_choices = torch.cat([prev_out, prev_tgt], dim=1)  # [B, 2]
    batch_size = prev_choices.size(0)
    prob = Bernoulli(torch.tensor([epsilon]*batch_size).unsqueeze(1))
    # sampling
    sample = prob.sample().long().to(prev_tgt.device)
    next_inp = torch.gather(prev_choices, 1, sample)
    return next_inp


def cal_epsilon(k, step, method):
    """
    Reference:
        Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks
        See details in https://arxiv.org/pdf/1506.03099.pdf
    """
    assert method in ['inv_sigmoid', 'exp', 'teacher_forcing']

    if method == 'exp':
        return k**step
    elif method == 'inv_sigmoid':
        return k/(k+math.exp(step/k))
    else:
        return 1.


def greedy_decode(model: Im2LatexModelTransformer, imgs, max_len, device):
    imgs = imgs.to(device)
    memory = model.encode(image_batch=imgs)
    ys = torch.ones(1, 1).fill_(START_TOKEN).type(torch.long).to(device)
    for _ in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(device)
        
        out = model.decode(ys, memory, tgt_mask, ys==PAD_TOKEN)
        # out = out.transpose(0, 1)
        prob = (out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(imgs.data).fill_(next_word)], dim=0)
        if next_word == END_TOKEN:
            break
    
    return ys
