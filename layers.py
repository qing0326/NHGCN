import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
import os


class GraphConvolution_adj(Module):
    """
    simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True): # 3840，32
        super(GraphConvolution_adj, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.channelMap=0
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
        if bias:
            self.bias = Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        output = torch.matmul(x, self.weight)-self.bias
        self.channelMap=output
        output = F.relu(torch.matmul(adj, output))
        return output

class GraphConvolution(Module):
    """
    simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):  # 3840，32
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
        if bias:
            self.bias = Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        output = torch.matmul(x, self.weight) - self.bias
        return output

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class ScaledTanh(nn.Module):
    def forward(self, x):
        return 1.5 * torch.tanh(x)

class BipolarSigmoid(nn.Module):
    def forward(self, x):
        return 2 * torch.sigmoid(x) - 1


class PowerLayer(nn.Module):
    '''
    The power layer: calculates the log-transformed power of the data
    '''
    def __init__(self, dim, length, step):
        super(PowerLayer, self).__init__()
        self.dim = dim
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))
        # self.pooling= nn.MaxPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x):
        return torch.log(self.pooling(x.pow(2)))


class Aggregator():

    def __init__(self, idx_area):
        # chan_in_area: a list of the number of channels within each area 每个区域内频道数量的列表
        self.chan_in_area = idx_area
        self.idx = self.get_idx(idx_area)
        self.area = len(idx_area)

    def forward(self, x):
        # x: batch x channel x data
        data = []
        for i, area in enumerate(range(self.area)):
            if i < self.area - 1:
                #
                data.append(self.aggr_fun(x[:, self.idx[i]:self.idx[i + 1], :], dim=1))
            else:
                data.append(self.aggr_fun(x[:, self.idx[i]:, :], dim=1))
        return torch.stack(data, dim=1)

    def get_idx(self, chan_in_area):
        idx = [0] + chan_in_area
        idx_ = [0]
        for i in idx:
            idx_.append(idx_[-1] + i)
        return idx_[1:]

    def aggr_fun(self, x, dim):
        # return torch.max(x, dim=dim).values
        return torch.mean(x, dim=dim)

class Conv2dWithRegularizers(nn.Conv2d):
    def __init__(self, *args, doRegularizers=True, l2_alpha=0.005, **kwargs):
        self.doRegularizers = doRegularizers
        self.l2_alpha = l2_alpha
        super(Conv2dWithRegularizers, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doRegularizers:
            self.weight.grad.data.add_(self.l2_alpha * self.weight.data)
        return super(Conv2dWithRegularizers, self).forward(x)

class channelAttention(nn.Module):
    def __init__(self, in_channels):
        in_channels=int(in_channels)
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, int(in_channels // 2),kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(int(in_channels // 2), in_channels,kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)
        z = self.Conv_Squeeze(z)
        z = self.Conv_Excitation(z)
        freqAtten = z
        freqAtten = torch.squeeze(freqAtten, 2)
        z = self.norm(z)
        # In addition, return to freqAtten for visualization
        return U * z.expand_as(U), freqAtten

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        x_reshaped = x.contiguous().view(-1,x.size(-2),x.size(-1))
        y,(h,c) = self.module(x_reshaped)

        if self.batch_first:
            y = y.contiguous().view(x.size(0), x.size(1), x.size(2), y.size(-1))
        else:
            y = y.contiguous().view(x.size(1), x.size(0), x.size(2), y.size(-1))
        return y

class TimeDistributedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, bidirectional=True):
        super(TimeDistributedLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.time_distributed_lstm = TimeDistributed(self.lstm, batch_first=True)

    def forward(self, x):
        return self.time_distributed_lstm(x)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
    def forward(self, q, k, v):
        attn = torch.matmul(q.transpose(1, 2) / self.temperature, k)  ### W*H  * H*W
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(v,attn)  #   H*W * W*W  得到W个权重
        return output