from layers import *

class NHGCN(nn.Module):    #严格保证通道独立
    def __init__(self,args, num_classes, input_size, sampling_rate,
                 dropout_rate, pool, pool_step,subject):
        # input_size: EEG frequency x channel x datapoint
        super(NHGCN, self).__init__()
        self.args=args
        self.pool = pool
        self.channel = int(input_size[1])
        self.subject=subject
        self.device = torch.device("cuda:{}".format(self.args.gpu))

        self.Depthwisconv1 = nn.Sequential(
            nn.ConstantPad2d((0, 63, 0, 0), 0),
            nn.Conv2d(self.channel, self.channel, (1, int(sampling_rate * 0.5)), padding=0, groups=self.channel))
        self.Depthwisconv2 = nn.Sequential(
            nn.ConstantPad2d((0, 31, 0, 0), 0),
            nn.Conv2d(self.channel, self.channel, (1, int(sampling_rate * 0.25)), padding=0, groups=self.channel))
        self.Depthwisconv3 = nn.Sequential(
            nn.ConstantPad2d((0, 15, 0, 0), 0),
            nn.Conv2d(self.channel, self.channel, (1, int(sampling_rate * 0.125)), padding=0, groups=self.channel))

        self.OneXOneConvWeight = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=(1, 1), stride=(1, 1)),
            nn.Tanh())
        self.activ = torch.nn.LeakyReLU()
        self.BN1 = nn.BatchNorm2d(self.channel)
        self.avgPool1 = PowerLayer(dim=-1, length=pool, step=int(pool_step))
        #LGGNet
        size = self.get_size_temporal(input_size)
        self.global_adj = nn.Parameter(torch.FloatTensor(self.channel, self.channel), requires_grad=True)
        nn.init.xavier_uniform_(self.global_adj)
        self.bn_1 = nn.BatchNorm1d(self.channel)
        self.GCN=GraphConvolution_adj(size[-1],self.args.outGraph_1)
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(int( self.channel*self.args.outGraph_1 ),num_classes))

    def forward(self,x):
        x = torch.unsqueeze(x, 2)
        out=x

        out1=self.Depthwisconv1(out)
        out2=self.Depthwisconv2(out)
        out3=self.Depthwisconv3(out)

        out = torch.cat((out1, out2, out3), dim=2)
        weight=self.OneXOneConvWeight(out)
        out=out*weight
        out = self.BN1(out)
        out = self.activ(out)
        out = self.avgPool1(out)
        out = out.view(out.size(0), out.size(1), -1)

        # adj全局邻接矩阵
        adj = self.get_adj(out)
        out = self.GCN(out,adj)

        out = self.bn_1(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

    def get_size_temporal(self,input_size):
        data = torch.ones((2,input_size[1], input_size[2]))
        data = torch.unsqueeze(data, 2)
        out=data

        out1=self.Depthwisconv1(out)
        out2=self.Depthwisconv2(out)
        out3=self.Depthwisconv3(out)
        out = torch.cat((out1, out2, out3), dim=2)
        out = self.activ(out)
        out = self.BN1(out)
        out = self.avgPool1(out)
        out = out.view(out.size(0), out.size(1), -1)

        size = out.size()
        return size


    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def get_adj(self, x, self_loop=True):
        adj = self.self_similarity(x)
        num_nodes = adj.shape[-1]
        adj = F.relu(adj * (self.global_adj + self.global_adj.transpose(1, 0)))
        if self_loop:
            DEVICE = torch.device("cuda:{}".format(self.args.gpu))
            adj = adj + torch.eye(num_nodes).to(DEVICE)
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj

    def self_similarity(self, x):
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)
        return s
