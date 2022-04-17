from curses import keyname
from os.path import join
import math

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class Im2LatexDataset(Dataset):
    def __init__(self, data_dir, split, args):
        """args:
        data_dir: root dir storing the prepoccessed data
        split: train, validate or test
        """
        assert split in ["train", "validate", "test"]
        self.args = args
        self.data_dir = data_dir
        self.split = split
        self.max_len = self.args.max_len
        self.pairs = self._load_pairs()
        if self.args.augment:
            self.transforms = torch.nn.Sequential(
                T.ColorJitter(0.5, 0.5, 0.5, 0.5),
                T.RandomInvert(p=0.5),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                T.RandomRotation((-10, 10)),
            )

    def _load_pairs(self):
        pairs = torch.load(join(self.data_dir, "{}.pkl".format(self.split)))
        for i, (img, formula) in enumerate(pairs):
            pair = (img, " ".join(formula.split()[:self.max_len]))
            pairs[i] = pair
        return pairs

    def _resize_img(self, img):
        CUTOFF = 21600
        shape = img.shape
        
        if shape[1]*shape[2] < CUTOFF:
            return img
        
        percent = math.sqrt(CUTOFF / (shape[1]*shape[2]))
        
        resize = T.Resize([int(shape[i]*percent) for i in shape])
        return resize(img)

    def __getitem__(self, index):
        if self.args.augment:
            img, formula = self.pairs[index]
            img = self._resize_img(img)
            return (self.transforms(img), " ".join(formula.split()[:self.max_len]))
        else:
            return self.pairs[index]

    def __len__(self):
        return len(self.pairs)
