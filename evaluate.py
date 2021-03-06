# load checkpoint and evaluating
import os
from os.path import join
from functools import partial
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Im2LatexDataset
from build_vocab import Vocab, load_vocab
from model.model_transformer import Im2LatexModelTransformer
from model.model import Im2LatexModel
from utils import collate_fn, collate_transformer_fn
from model.decoding_updated import LatexProducerUpdated
from model.score import score_files


def main():
    parser = argparse.ArgumentParser(description="Im2Latex Evaluating Program")
    parser.add_argument('--model_path', required=True,
                        help='path of the evaluated model')
    parser.add_argument('--model_type', required=True,
                        help='type of model to evaluate')

    # model args
    parser.add_argument("--augment", action='store_true',
                        default=False, help="Perform data augmentation")
    parser.add_argument("--data_path", type=str,
                        default="./data/", help="The dataset's dir")
    parser.add_argument("--cuda", action='store_true',
                        default=True, help="Use cuda or not")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--max_len", type=int,
                        default=150, help="Max step of decoding")
    parser.add_argument("--split", type=str,
                        default="test", help="The data split to decode")

    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        print("MODEL NOT FOUND")


    # load checkpoint
    checkpoint = torch.load(join(args.model_path, 'best_ckpt.pt'), map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model_args = checkpoint['args']
    
    # load vocab
    vocab = load_vocab(args.data_path)
    use_cuda = True if args.cuda and torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")
    # load dataloader
    if args.model_type == 'transformer':
        data_loader = DataLoader(
            Im2LatexDataset(args.data_path,'test' , args),
            batch_size=args.batch_size,
            collate_fn=partial(collate_transformer_fn, vocab.sign2id),
            pin_memory=True if use_cuda else False
        )
        model = Im2LatexModelTransformer(model_args)
    else:
        data_loader = DataLoader(
            Im2LatexDataset(args.data_path,'test' , args),
            batch_size=args.batch_size,
            collate_fn=partial(collate_fn, vocab.sign2id),
            pin_memory=True if use_cuda else False
        )
        model = Im2LatexModel(
            model_args,
            len(vocab), model_args.emb_dim,
            model_args.enc_rnn_h, model_args.dec_rnn_h,
            'device', dropout=model_args.dropout,
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    result_path = os.path.join(args.model_path, 'predicted.txt')
    ref_path = os.path.join(args.model_path, 'reference.txt')
    result_file = open(result_path, 'w')
    ref_file = open(ref_path, 'w')

    latex_producer = LatexProducerUpdated(
        model, vocab, max_len=args.max_len,
        use_cuda=use_cuda, beam_size=args.beam_size, )

    if args.model_type == 'transformer':
        for imgs, (tgt4training, tgt4cal_loss) in tqdm(data_loader):
            reference = latex_producer._idx2formulas(tgt4cal_loss)
            results = latex_producer(imgs)
            # except RuntimeError:
            #     print('ERROR in evaluate')
            #     break

            result_file.write('\n'.join(results))
            ref_file.write('\n'.join(reference))
    else:
        for imgs, tgt4training, tgt4cal_loss in tqdm(data_loader):
            reference = latex_producer._idx2formulas(tgt4cal_loss)
            results = latex_producer(imgs)
            # except RuntimeError:
            #     print('ERROR in evaluate')
            #     break

            result_file.write('\n'.join(results))
            ref_file.write('\n'.join(reference))



    result_file.close()
    ref_file.close()
    score = score_files(result_path, ref_path)
    print("beam search result: ", score)


if __name__ == "__main__":
    main()
