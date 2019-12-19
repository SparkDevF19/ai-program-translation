# Implementation from https://github.com/dmlc/dgl/blob/master/examples/pytorch/tree_lstm/tree_lstm.py
import sys
import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl

#in_dim =
#h_dim =
# num_layers = 1

# tree LSTM cell for binary trees
class BinaryTreeLSTMCell(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.iou_x = nn.Linear(in_dim, h_dim * 3)         # i, o, u matrices for x (cell state)
        self.iou_hl = nn.Linear(h_dim, h_dim * 3)         # i, o, u matrices for left h (hidden state)
        self.iou_hr = nn.Linear(h_dim, h_dim * 3)         # i, o, u matrices for right h (hidden state)
        self.f_x = nn.Linear(in_dim, h_dim)               # forget for x

        # forget for hidden state
        self.f_h = nn.ModuleList([[nn.Linear(h_dim, h_dim), nn.Linear(h_dim, h_dim)],
                                [nn.Linear(h_dim, h_dim), nn.Linear(h_dim, h_dim)]])

    # takes in input, cell states, and hidden states
    def forward(self, x, hl, hr, cl, cr):
        # i, o, u, gates
        self.iou = self.iou_x(x) + self.iou_hl(hl) + self.iou_hr(hr)

        # split
        i, o, u = th.split(iou, iou.size(1) // 3, dim=1)      

        # apply activation functions
        i = F.sigmoid(i)                         
        o = F.sigmoid(o)
        u = F.tanh(u)

        # forget for left and right
        fl = F.sigmoid(self.f_x(x) + self.f_h[0][0](hr) + self.f_h[0][1](hl))
        fr = F.sigmoid(self.f_x(x) + self.f_h[1][0](hr) + self.f_h[1][1](hl))

        # calculate hidden state and cell state
        c = i * u + fl * cl + fr * cr
        h = o * F.tanh(c)

        # return hidden state and cell state
        return h, c

# TreeLSTM encoder
class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, embedding_size):
        super().__init__()
        self.initial_h = 0
        self.initial_c = 0
        self.in_dim = in_dim
        self.h_dim = h_dim

        # Dropout Layer (may be useful) (TODO:tune hyperparameters)
        self.drop = nn.Dropout(p=0.5, inplace=False)

        # Binary LSTM cell and embedding layer
        self.tree_cell = BinaryTreeLSTMCell(in_dim, h_dim)
        self.embed = nn.Embedding(in_dim, embedding_size)

    # compute embeddings for source tree and subtrees
    def forward(self, batch):
        # 
        binary_cell = BinaryTreeLSTMCell(in_dim, h_dim)

        # iterate through each tree in batch
        for tree in batch:
            # hidden state
            hr = 0
            hl = 0

            # cell state
            cr = 0
            cl = 0

            # iterate postorder over the tree, passing each layer to the lstm cell
            current = 0
            nodes_stack = []

            while(True):
                # while root is not empty
                while (tree.successors(current).size() != 0):
                    nodes_stack.append(tree.successors(current[1]))
                    nodes_stack.append(current)
                
                current = tree.successors(current)[0]
                current = nodes_stack.pop()
                
                if (tree.successors.size() != 1 and tree.successors(current)[1] in nodes_stack):
                    nodes_stack.pop()
                    nodes_stack.append(current)
                    current = tree.successors(current)[1]

                else:
                    # run binary lstm for node
                    x = tree[current].ndata['info']
                    h, c = binary_cell.forward(x, hl, hr, cl, cr)
                    tree[current].ndata['e'] = self.embed(x)
                    tree[current].ndata['h'] = h
                    tree[current].ndata['c'] = c

                # stack is empty
                if (len(nodes_stack) == 0):
                    break


            '''
            # base case (LOOK AT SYNTAX)
            if (tree[0].successors(0).ndata['info'][0]== None):
                return
            # compute embeddings 
            else:
                x = tree[node_initial].ndata['info']
                hl, cl = binary_cell.forward()
                hr, cr = binary_cell.forward()
                binary_cell.forward(x, hl, hr, cl, cr)'''



# Attention class to locate the source sub-tree
class Attention(nn.Module):
    def __init__(self, h_dim):
        # Weights matrices of size d * d (d is the embedding dimension)
        self.h_dim = h_dim
        W_0 = nn.Linear(h_dim, h_dim)
        W_1 = nn.Linear(h_dim, h_dim)
        W_2 = nn.Linear(h_dim, h_dim)
        
    # get the source tree
    def forward(self, tree, h_t):
        # calculate probability while doing post-order traversal through tree
        current = 0
        nodes_stack = []

        while(True):
            
            # expectation
            e_s = th.zeros(h_dim)
            
            # while root is not empty
            while (tree.successors(current).size() != 0):
                nodes_stack.append(tree.successors(current[1]))
                nodes_stack.append(current)
                
                current = tree.successors(current)[0]
                current = nodes_stack.pop()
                
                if (tree.successors.size() != 1 and tree.successors(current)[1] in nodes_stack):
                    nodes_stack.pop()
                    nodes_stack.append(current)
                    current = tree.successors(current)[1]

                else:
                    # calculate probability
                    p = th.exp(tree[current].ndata['h'].transpose() * self.W_0(h_t))
                    
                    # compute expectation of h_t to be throughout all the nodes in the tree
                    e_s += tree[current].ndata['h'] * p

                if (len(nodes_stack) == 0):
                    break

            # compute e_t by combining W_1, W_2, e_s, and h_t and pass through activation function tanh
            e_t = F.tanh(self.W_1(e_s) + self.W_2(h_t))

            return e_t

        

# Decoder generates the target tree starting from a single root node
class Decoder(nn.Module):
    def __init__(self, e_t, h_dim, vocab_size):
        super().__init__()  
        # trainable matrix of vocab size of outputs and embedding dimension
        self.W_tt = nn.Linear(h_dim, vocab_size)
        self.B_t = nn.Linear(h_dim, vocab_size)
        
        # attention mechanism
        self.attention = Attention(h_dim)
        
    # generate target tree from source tree
    def forward(self, batch):
        for tree in batch:
            # make tree with one node
            target_tree = dgl.DGLGraph(1)
            
            # copy LSTM state from encoder of root of source tree and attach to root of target tree until empty list
            target_tree[0].ndata['h'] = tree[0].ndata['h']
            
            # initialize expanding node queue
            nodes_queue = [0]
            current = 0

            # stop if there are no nodes left to expand
            while (nodes_queue):
                # current node is the first one in queue
                current = nodes_queue.pop(0)

                # compute e_t
                e_t = attention.forward(tree, target_tree[current].ndata['h'])

                # feed it into softmax regression network to get our token
                t_t = th.max(F.softmax(W_tt(e_t)))
                
                # if t_t isn't EOS, make two children nodes
                if (t_t != "EOS"):
                    # make two children
                    target_tree.add_nodes(2)
                    target_tree([current, current], [len(target_tree) - 1, len(target_tree) - 2])
                    
                    # add children to queue
                    nodes_queue.append(target_tree[current].successors()[0])
                    nodes_queue.append(target_tree[current].successors()[1])


class TreeToTreeLSTM(nn.Module):    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = Encoder
        self.decoder = Decoder

    def forward(self, batch):
        encoder.forward(batch)
        decoder.forward(batch)

    