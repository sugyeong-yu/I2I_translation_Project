import torch
import os
import random
from torch.utils import data

class CelebA(data.Dataset):
    def __init__(self,img_dir,attr_path,selected_attrs,transform,mode):
        self.img_dir=img_dir
        self.attr_path=attr_path
        self.selected_attrs=selected_attrs
        self.transform=transform
        self.mode=mode
        self.train_dataset=[]
        self.test_dataset=[]
        self.attr2idx= {}
        self.idx2attr={}
        self.preprocess()

        if mode=='train':
            self.num_images=len(self.train_dataset)
        else:
            self.num_images=len(self.test_dataset)
    def preprocess(self):
        # CelebA는 preprocessing이 필요
        lines = [line.rstrip() for line in open(self.attr_path,'r')]
        all_attr_names=lines[1].split()
        for i,attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i]=attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
