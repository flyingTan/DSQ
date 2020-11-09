import os
import pickle

import cv2
import nori2 as nori
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from meghair.utils.imgproc import imdecode
from neupeak.dataset2 import BaseDataset
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

# np.random.seed(0)

nf = nori.Fetcher()


class Imagenet(Dataset):
    def __init__(self, dataset_name, trans):
        assert dataset_name in ("train", "val")
        root_dir = "/data/workspace/face/dataset/imagenet/"
        self.dataset_name = dataset_name
        self.transforms = trans

        nfname = {"train": "imagenet.train.nori.list", "val": "imagenet.val.nori.list",}[dataset_name]

        self.nr_class = 1000
        self.nid_filename = os.path.join(root_dir, nfname)
        self.load()

    def load(self):
        self.nid_labels = []
        self.targets = []
        # e.g. 1261,a450007d6bfdfb 0 n01440764/n01440764_1383.JPEG
        if self.dataset_name == "train":
            with open(self.nid_filename) as f:
                for line in f:
                    nid, label, filename = line.strip().split("\t")
                    filename = filename.split("/")[1]
                    #  lb = torch.LongTensor((int(label),))
                    lb = int(label)
                    self.targets.append(lb)
                    self.nid_labels.append((nid, lb, filename))
        else:
            with open(self.nid_filename) as f:
                for line in f:
                    nid, label, _ = line.strip().split("\t")
                    # lb = torch.LongTensor((int(label),))
                    lb = int(label)
                    self.targets.append(lb)
                    self.nid_labels.append((nid, lb, _))
        return self

    def __getitem__(self, idx):
        nid, label, _ = self.nid_labels[idx]
        data = nf.get(nid)
        img = imdecode(data)[:, :, :3]
        img = self.transforms(img)
        label = torch.LongTensor([label])
        return img, label

    def __len__(self):
        return len(self.nid_labels)



if __name__ == "__main__":
    import time
    from torch.utils.data import DataLoader

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = Imagenet(
        "train",
        transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    train_loader = DataLoader(train_dataset, batch_size=64 * 8, shuffle=True, num_workers=4, pin_memory=True)

    t_begin = time.time()
    i = 0
    for img, label in train_loader:
        #    label = label.squeeze()
        i += 1
    print("total time is {}".format(time.time() - t_begin))
    print(train_dataset.__len__())
    print(i)
