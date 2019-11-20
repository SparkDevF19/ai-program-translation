# Implementation from https://github.com/dmlc/dgl/blob/master/examples/pytorch/tree_lstm/tree_lstm.py
import sys
import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# tree LSTM cell for binary trees
class BinaryTreeLSTMCell(nn.Module):
  def __init__(self, in_dim, h_dim):
    super().__init__()
    iou_x = nn.Linear(in_dim, h_dim * 3)      # i, o, u matrices for x
    iou_h = nn.Linear(h_dim, h_dim * 3)       # i, o, u matrices for h
    f_x = nn.Linear(in_dim, h_dim)            # forget for x
    lf_h = nn.Linear(h_dim, h_dim)            # forget for left nodes' hidden state
    rf_h = nn.Linear(h_dim, h_dim)            # forget for right node's hidden state


  def forward(self, x, lh, rh):
    i, o, u = th.split(ioux, 3, )             # split 
    i = th.sigmoid(i)                         
    o = th.sigmoid(o)
    u = th.tanh(u)

example = TreeLSTMCell(512, 512, 2)

print(example.layers[0])

# Attetion class

# decoder class

# tree-to-tree class
class TreeToTree(nn.Module)
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder