# Implementation from https://github.com/dmlc/dgl/blob/master/examples/pytorch/tree_lstm/tree_lstm.py
import sys
import os
import torch.nn as nn
from .treelstm import TreeLSTM, Tree, BatchedTree
import json


# get data
with open('./data/preprocessed_progs_train.json') as data:
    loaded_data = json.load(data)
    # TODO do some stuff here

"""
# train network
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
"""