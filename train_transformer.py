import os
import argparse
from functools import partial

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.tensorboard import SummaryWriter

from model.model_transformer import Im2LatexModelTransformer 
from model.training import Transformer_Trainer

from utils import collate_transformer_fn, get_checkpoint
from data import Im2LatexDataset
from build_vocab import Vocab, load_vocab


def main():
    # get args
    parser = argparse.ArgumentParser(description="Im2Latex Training Program")
    # parser.add_argument('--path', required=True, help='root of the model')

    # model args
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
    
    # data args
    parser.add_argument("--data_path", type=str,
                        default="./data/", help="The dataset's dir")
    
    # training args
    parser.add_argument("--max_len", type=int,
                        default=150, help="Max size of formula")
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
                        "See details in https://arxiv.org/pdf/1506.03099.pdf"
                        )

    parser.add_argument("--lr_decay", type=float, default=0.5,
                        help="Learning Rate Decay Rate")
    parser.add_argument("--lr_patience", type=int, default=3,
                        help="Learning Rate Decay Patience")
    parser.add_argument("--clip", type=float, default=2.0,
                        help="The max gradient norm")
    parser.add_argument("--save_dir", type=str,
                        default="./ckpts", help="The dir to save checkpoints")
    parser.add_argument("--print_freq", type=int, default=100,
                        help="The frequency to print message")
    parser.add_argument("--seed", type=int, default=2020,
                        help="The random seed for reproducing ")
    parser.add_argument("--from_check_point", action='store_true',
                        default=False, help="Training from checkpoint or not")

    args = parser.parse_args()
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
    print("Construct data loader...")
    train_loader = DataLoader(
        Im2LatexDataset(args.data_path, 'test', args.max_len),
        batch_size=args.batch_size,
        collate_fn=partial(collate_transformer_fn, vocab.sign2id),
        pin_memory=True if use_cuda else False,
        num_workers=num_workers)
    val_loader = DataLoader(
        Im2LatexDataset(args.data_path, 'validate', args.max_len),
        batch_size=args.batch_size,
        collate_fn=partial(collate_transformer_fn, vocab.sign2id),
        pin_memory=True if use_cuda else False,
        num_workers=num_workers)

    # construct model
    print("Constructing model: ")
    vocab_size = len(vocab)
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
    # model = Im2LatexModel(
    #     vocab_size, args.emb_dim,
    #     args.enc_rnn_h, args.dec_rnn_h,
    #     device, dropout=args.dropout
    # )
    model = model.to(device)
    print("Model Summary:")
    print(model)

    import code
    code.interact(local=locals())
    # construct optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=args.lr_decay,
        patience=args.lr_patience,
        verbose=True,
        min_lr=args.min_lr)

    writer = SummaryWriter()
    
    if from_check_point:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        lr_scheduler.load_state_dict(checkpoint['lr_sche'])
        # init trainer from checkpoint
        trainer = Transformer_Trainer(optimizer, model, lr_scheduler,
                          train_loader, val_loader, args,
                          use_cuda=use_cuda,
                          init_epoch=epoch, last_epoch=max_epoch,
                          writer=writer)
    else:
        trainer = Transformer_Trainer(optimizer, model, lr_scheduler,
                          train_loader, val_loader, args,
                          use_cuda=use_cuda,
                          init_epoch=1, last_epoch=args.epoches,
                          writer=writer)
    # begin training
    print('Beginning training...')
    trainer.train()


if __name__ == "__main__":

    main()
