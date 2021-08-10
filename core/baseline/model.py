import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class CNN7(nn.Module):
    def __init__(self,
                 x_dim = [1,28,28],    # input dimension
                 k_size = 3,            # kernel size
                 c_dims = [32,64],      # conv channel dimensions
                 p_sizes = [2,2],        # pooling sizes
                 h_dims = [128],        # hidden dimensions
                 y_dim = 10,           # output dimension
                 USE_BN = True
                 ):
        super(CNN7, self).__init__()
        self.x_dim = x_dim
        self.k_size = k_size
        self.c_dims = c_dims
        self.p_sizes = p_sizes
        self.h_dims = h_dims
        self.y_dim = y_dim
        self.USE_BN = USE_BN

        self.build_graph()
        self.init_param()

    def build_graph(self):
        self.layers = []
        # Conv layers
        prev_c_dim = self.x_dim[0]  # input channel
        for i, c_dim in enumerate(self.c_dims):
            self.layers.append(
                nn.Conv2d(
                    in_channels=prev_c_dim,
                    out_channels=c_dim,
                    kernel_size=self.k_size,
                    stride=1,
                    padding=1
                )  # conv
            )
            if self.USE_BN:
                self.layers.append(
                    nn.BatchNorm2d(num_features=c_dim)
                )
            self.layers.append(nn.ReLU())
            if i % 2 == 1:
                self.layers.append(
                    nn.MaxPool2d(kernel_size=self.p_sizes[i//2], stride=self.p_sizes[i//2])
                )
            # self.layers.append(nn.Dropout2d(p=0.1))  # p: to be zero-ed
            prev_c_dim = c_dim
            # Dense layers
        self.layers.append(nn.Flatten())
        p_prod = np.prod(self.p_sizes)
        prev_h_dim = prev_c_dim * (self.x_dim[1]//p_prod) * (self.x_dim[2]//p_prod)
        for h_dim in self.h_dims:
            self.layers.append(
                nn.Linear(
                    in_features=prev_h_dim,
                    out_features=h_dim,
                    bias=True
                )
            )
            self.layers.append(nn.ReLU(True))  # activation
            self.layers.append(nn.Dropout2d(p=0.1))  # p: to be zero-ed
            prev_h_dim = h_dim

        self.layers.append(nn.Linear(256, self.y_dim))

        # Concatanate all layers
        self.net = nn.Sequential()
        for l_idx, layer in enumerate(self.layers):
            layer_name = "%s_%02d" % (type(layer).__name__.lower(), l_idx)
            self.net.add_module(layer_name, layer)

    def forward(self, x):
        out = self.net(x)
        return out  # mu:[N x K x D] / pi:[N x K] / sigma:[N x K x D]

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # init conv
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):  # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

def call_bn(bn, x):
    return bn(x)

class CNN3(nn.Module):
    def __init__(self, input_channel=1, n_outputs=10, dropout_rate=0.25, momentum=0.1):
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        super(CNN3, self).__init__()
        self.c1=nn.Conv2d(input_channel, 32,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(32,64,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1)
        self.linear1=nn.Linear(1152, 128)
        self.linear2=nn.Linear(128, 64)
        self.linear3=nn.Linear(64, n_outputs)

    def forward(self, x,):
        h=x
        h=F.relu(self.c1(h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.relu(self.c2(h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.relu(self.c3(h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h = h.view(h.size(0), -1)
        h=F.relu(self.linear1(h))
        h=F.relu(self.linear2(h))
        logit=self.linear3(h)
        return logit
        
    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # init conv
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):  # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)