import os
import json

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pdb


def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    # criterion = nn.CrossEntropyLoss()
    # loss = criterion(logits, labels)
    loss *= labels.size(1)
    return loss


def cosine_sim(im, s):
    return im.mm(s.t())


def find_tail_by_KG(facts, h_id, r_id):
    if h_id in facts.keys():
        if r_id in facts[h_id].keys():
            return facts[h_id][r_id]
    return None
