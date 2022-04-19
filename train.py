import os
import argparse
from functools import partial

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.tensorboard import SummaryWriter

from model import Im2LatexModel
from model.training import LSTMTrainer, TransformerTrainer
from model.model_transformer import Im2LatexModelTransformer
from utils import collate_fn, collate_transformer_fn, get_checkpoint
from data import Im2LatexDataset
from build_vocab import Vocab, load_vocab


def next_nonexistent_dir(d):
    i = 0
    d_new = d
    while os.path.exists(d_new):
        i += 1
        d_new = '%s_%i' % (d, i)
    return d_new

def get_tensorboard_writer(experiment, verbose=True):
    log_dir = next_nonexistent_dir("./runs/experiment{}".format(experiment))
    os.mkdir(log_dir)
    if verbose:
        print("\nLogging directory:", log_dir, "\n")
    return SummaryWriter(log_dir=log_dir), log_dir


def main():

    # get args
    parser = argparse.ArgumentParser(description="Im2Latex Training Program")
    # parser.add_argument('--path', required=True, help='root of the model')
    
    # row_encoder: Whether to have the row encoder or not, if not, feed directly from CNN to decoder
    # paper_emb: Fixed embeddings applied as the encoder initial hidden state
    # new_emb: Embeddings applied to the output of the CNN or row encoder (depending on row_encoder)
    experiments = {        
        
        # Row encoder, paper_emb - CLOSEST TO THE PAPER
        1: {"use_transformer": False, "row_encoder": True, "paper_emb": True, "new_emb": False},
        
        # Transformer, 6+6 layers, use patching
        2: {"use_transformer": True, "size": "small", "use_patches": True, "patch_size":2},
        
        # Row encoder, new_emb
        3: {"use_transformer": False, "row_encoder": True, "paper_emb": False, "new_emb": True},
        
        # No row encoder, new_emb
        4: {"use_transformer": False, "row_encoder": False, "paper_emb": False, "new_emb": True},
        
        # Transformer, 6+6 layers, no patching
        5: {"use_transformer": True, "size": "small", "use_patches": False},
        
        # Row encoder, no embeddings
        6: {"use_transformer": False, "row_encoder": True, "paper_emb": False, "new_emb": False},
        
        # Transformer, 8+8 layers, no patching
        7: {"use_transformer": True, "size": "medium", "use_patches": False},
        
        # Row encoder, paper_emb, new_emb
        8: {"use_transformer": False, "row_encoder": True, "paper_emb": True, "new_emb": True},
        
        # No row encoder, no embeddings
        9: {"use_transformer": False, "row_encoder": False, "paper_emb": False, "new_emb": False},
       
        # Transformer, 12+12 layers, use patching
        10: {"use_transformer": True, "size": "large", "use_patches": True, "patch_size": 2},

        # Transformer, 12+12 layers, use patching
        11: {"use_transformer": True, "size": "large", "use_patches": False},
        
        # Transformer, 8+8 layers, use patching
        12: {"use_transformer": True, "size": "medium", "use_patches": True, "patch_size": 2},
    }
    # experiment args
    parser.add_argument("experiment", type=int, help="Specify an experiment")
    parser.add_argument("--debug", action='store_true', default=False,
                        help="Debugging mode using small dataset")
    
    
    # LSTM encoder model args
    parser.add_argument("--emb_dim", type=int,
                        default=80, help="Embedding size")
    parser.add_argument("--enc_rnn_h", type=int, default=256,
                        help="The hidden state of the row encoder RNN")
    parser.add_argument("--dec_rnn_h", type=int, default=512,
                        help="The hidden state of the decoder RNN")
    parser.add_argument("--data_path", type=str,
                        default="./data/", help="The dataset's dir")
    
    
    # transformer model args
    
    parser.add_argument("--use_row_embeddings", type=bool,
                        default=True, help="weather or not to use row positional embeddings")
    parser.add_argument("--weight_decay", type=float,
                        default=0.01, help="Optimizer weight decay")
    parser.add_argument("--feat_size", type=int,
                        default=512, help="Feature_Embedding size")
    parser.add_argument("--encoder_nheads", type=int, default=8,
                        help="Number of heads of transormer encoder")
    parser.add_argument("--encoder_dim_feedforward", type=int, default=1024,
                        help="Hidden Dimension for encoder fc layer")
    parser.add_argument("--encoder_num_layers", type=int, default=6,
                        help="Number of encoder attention layers")
    parser.add_argument("--encoder_dropout", type=float, default=0.2,
                        help="Dropout to be used for encoder")
    parser.add_argument("--encoder_max_sequence_length", type=int, default=2048,
                        help="Max output sequence length expected for encoder input")
    
    parser.add_argument("--decoder_nheads", type=int, default=8,
                        help="Number of decoder heads for each layer")
    parser.add_argument("--decoder_dim_feedforward", type=int, default=1024,
                        help="Hidden dimension for decoder fc layer")
    parser.add_argument("--decoder_num_layers", type=int, default=6,
                        help="Number of decoder attention layers")
    parser.add_argument("--decoder_dropout", type=float, default=0.2,
                        help="Dropout to be used for decoder")
    parser.add_argument("--decoder_max_sequence_length", type=int, default=1024,
                        help="Max output sequence length expected for decoder output")
    
    # training args
    parser.add_argument("--augment", action='store_true',
                        default=False, help="Perform data augmentation")
    parser.add_argument("--max_len", type=int,
                        default=200, help="Max size of formula")
    parser.add_argument("--dropout", type=float,
                        default=0., help="Dropout probility")
    parser.add_argument("--cuda", action='store_true',
                        default=True, help="Use cuda or not")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoches", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning Rate")
    parser.add_argument("--min_lr", type=float, default=3e-5,
                        help="Learning Rate")
    parser.add_argument("--sample_method", type=str, default="teacher_forcing",
                        choices=('teacher_forcing', 'exp', 'inv_sigmoid'),
                        help="The method to schedule sampling")
    parser.add_argument("--decay_k", type=float, default=1.,
                        help="Base of Exponential decay for Schedule Sampling. "
                        "When sample method is Exponential deca;"
                        "Or a constant in Inverse sigmoid decay Equation. "
                        "See details in https://arxiv.org/pdf/1506.03099.pdf")
    parser.add_argument("--lr_decay", type=float, default=0.5,
                        help="Learning Rate Decay Rate")
    parser.add_argument("--lr_patience", type=int, default=3,
                        help="Learning Rate Decay Patience")
    parser.add_argument("--clip", type=float, default=2.0,
                        help="The max gradient norm")
    parser.add_argument("--save_dir", type=str,
                        help="The dir to save checkpoints")
    parser.add_argument("--print_freq", type=int, default=100,
                        help="The frequency to print message")
    parser.add_argument("--seed", type=int, default=2020,
                        help="The random seed for reproducing ")
    parser.add_argument("--from_check_point", action='store_true',
                        default=False, help="Training from checkpoint or not")
    
    args = parser.parse_args()
    
    # Args preprocessing
    writer, log_dir = get_tensorboard_writer(args.experiment)
    
    if args.save_dir == None:
        args.save_dir = "./ckpts/ckpts_" + log_dir[len("./runs/"):]
    
    print("\nCheckpoint directory:", args.save_dir, "\n")
    
    if args.augment:
        print("Using data augmentation.")
    
    # Run training
    if experiments[args.experiment]["use_transformer"]:
        TrainTransformer(experiments, args, writer)
    else:
        TrainLSTMEncoder(experiments, args, writer)

   
def TrainTransformer(experiments, args, writer):
    max_epoch = args.epoches
    from_check_point = args.from_check_point
    if from_check_point:
        checkpoint_path = get_checkpoint(args.save_dir)
        checkpoint = torch.load(checkpoint_path)
        args = checkpoint['args']
    print("Training args:", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Building vocab
    print("Load vocab...")
    vocab = load_vocab(args.data_path)

    use_cuda = True if args.cuda and torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device:", device)

    # Multi-processing doesn't function well with Windows OS
    if os.name == 'nt':
        num_workers = 0
    else:
        num_workers = 4
    
    # data loader
    if args.debug:
        print('Running in debug mode')
    print("Construct transformer data loader...")
    train_loader = DataLoader(
        Im2LatexDataset(args.data_path,'test' if args.debug else 'train', args),
        batch_size=args.batch_size,
        collate_fn=partial(collate_transformer_fn, vocab.sign2id),
        pin_memory=True if use_cuda else False,
        num_workers=num_workers)
    val_loader = DataLoader(
        Im2LatexDataset(args.data_path, 'validate', args),
        batch_size=args.batch_size,
        collate_fn=partial(collate_transformer_fn, vocab.sign2id),
        pin_memory=True if use_cuda else False,
        num_workers=num_workers)

    # construct model
    print("Constructing model: ")
    vocab_size = len(vocab)
    if experiments[args.experiment]['size'] == 'small':
        print("Model type: small")
        args.encoder_num_layers = 6
        args.decoder_num_layers = 6
    elif experiments[args.experiment]['size'] == 'medium':
        print("Model type: medium")
        args.encoder_num_layers = 8
        args.decoder_num_layers = 8
    elif experiments[args.experiment]['size'] == 'large':
        print("Model type: large")
        args.encoder_num_layers = 12
        args.decoder_num_layers = 12

    if experiments[args.experiment]['use_patches']:
        args.__dict__.update({
            "use_patches": True,
            "patch_size":  experiments[args.experiment]['patch_size'] })
    else:
        args.__dict__.update({
            "use_patches": False})

    args.__dict__.update({"vocab_size": vocab_size})
    # config = Config(
    #     vocab_size=vocab_size,
    #     feat_size=512,
    #     encoder_nheads=8,
    #     encoder_dim_feedfoward=2048,
    #     encoder_num_layers=6,
    #     encoder_dropout=0.2,
    #     encoder_max_sequence_length=1024,
    #     decoder_nheads=8,
    #     decoder_dim_feedfoward=2048,
    #     decoder_num_layers=8,
    #     decoder_dropout=0.2,
    #     decoder_max_sequence_length=1024
    # )
    model = Im2LatexModelTransformer(args)
    model = model.to(device)
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    print("Model memory footprint: {}".format(mem))

    # print("Model Summary:")
    # print(model)
    # construct optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=args.lr_decay,
        patience=args.lr_patience,
        verbose=True,
        min_lr=args.min_lr
    )
    
    if from_check_point:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        lr_scheduler.load_state_dict(checkpoint['lr_sche'])
        # init trainer from checkpoint
        trainer = TransformerTrainer(optimizer, model, lr_scheduler,
                          train_loader, val_loader, args,
                          device, use_cuda=use_cuda,
                          init_epoch=epoch, last_epoch=max_epoch,
                          writer=writer)
    else:
        trainer = TransformerTrainer(optimizer, model, lr_scheduler,
                          train_loader, val_loader, args,
                          device, use_cuda=use_cuda,
                          init_epoch=1, last_epoch=args.epoches,
                          writer=writer)
    # begin training
    print('Beginning training for transformer...')
    trainer.train()



def TrainLSTMEncoder(experiments, args, writer):
    max_epoch = args.epoches
    from_check_point = args.from_check_point
    if from_check_point:
        checkpoint_path = get_checkpoint(args.save_dir)
        checkpoint = torch.load(checkpoint_path)
        args = checkpoint['args']
    print("Training args:", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Building vocab
    print("Load vocab...")
    vocab = load_vocab(args.data_path)

    use_cuda = True if args.cuda and torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device:", device)

    # Multi-processing doesn't function well with Windows OS
    if os.name == 'nt':
        num_workers = 0
    else:
        num_workers = 4

    # data loader
    if args.debug:
        print('Running in debug mode')
    print("Construct data loader...")
    train_loader = DataLoader(
        Im2LatexDataset(args.data_path, 'test' if args.debug else 'train', args),
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, vocab.sign2id),
        pin_memory=True if use_cuda else False,
        num_workers=num_workers)
    val_loader = DataLoader(
        Im2LatexDataset(args.data_path, 'validate', args),
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, vocab.sign2id),
        pin_memory=True if use_cuda else False,
        num_workers=num_workers)
    
    # construct model
    print("Construct model")
    vocab_size = len(vocab)
    model = Im2LatexModel(
        experiments[args.experiment],
        vocab_size, args.emb_dim,
        args.enc_rnn_h, args.dec_rnn_h,
        device, dropout=args.dropout,
    )
    
    model = model.to(device)
    print("Model Settings:")
    print(model)

    # construct optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=args.lr_decay,
        patience=args.lr_patience,
        verbose=True,
        min_lr=args.min_lr)

    if from_check_point:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        lr_scheduler.load_state_dict(checkpoint['lr_sche'])
        # init trainer from checkpoint
        trainer = LSTMTrainer(optimizer, model, lr_scheduler,
                          train_loader, val_loader, args,
                          device, use_cuda=use_cuda,
                          init_epoch=epoch, last_epoch=max_epoch,
                          writer=writer)
    else:
        trainer = LSTMTrainer(optimizer, model, lr_scheduler,
                          train_loader, val_loader, args,
                          device, use_cuda=use_cuda,
                          init_epoch=1, last_epoch=args.epoches,
                          writer=writer)
    # begin training
    print('Beginning training...')
    trainer.train()


if __name__ == "__main__":
    main()
