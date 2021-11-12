import os
import torch
import glob
from torch import optim
import numpy as np
import time
import argparse
from recognizer.models.encoder_vgg import Encoder
from load_data import NUM_WRITERS
from network_tro import ConTranModel
from load_data import loadData as load_data_func
from loss_tro import CER

lr_rec = 1 * 1e-5
rec_params = list(Encoder.parameters())
hidden_size_enc = hidden_size_dec = 512
IMG_HEIGHT = 128
IMG_WIDTH = 128
OOV = True
gpu = torch.device('cuda')

rec_opt = optim.Adam([p for p in rec_params if p.requires_grad], lr=lr_rec)

def all_data_loader():
    data_train, data_test = load_data_func(OOV)
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_THREAD, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREAD, pin_memory=True)
    return train_loader, test_loader

train_loader, test_loader = all_data_loader()
model = Encoder(hidden_size_enc, IMG_HEIGHT, IMG_WIDTH, True, None, False)

for train_data_list in train_loader:
    '''rec update'''
    rec_opt.zero_grad()
    l_rec_tr = model(train_data_list, epoch, 'rec_update', cer_tr)
    rec_opt.step()