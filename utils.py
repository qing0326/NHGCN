import time
import pprint
import random
from networks import *
from myNet import *
from eeg_dataset import *
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import torch
import numpy as np
from scipy.integrate import simps
import mne
from mne.time_frequency import psd_array_multitaper
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
import os

def set_gpu(x):
    torch.set_num_threads(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def get_model(args,subject):
    input_size = args.input_shape
    model = NHGCN(
        args=args, num_classes=args.num_class, input_size=input_size,
        sampling_rate=args.sampling_rate,
        dropout_rate=args.dropout,
        pool=args.pool, pool_step=args.pool_step,subject=subject)
    return model


def get_dataloader(data, label, batch_size):
    # load the data
    dataset = eegDataset(data, label)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return loader


def get_metrics(y_pred, y_true,classes):
    label = np.arange(classes)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred,average='macro')
    if classes is not None:
        cm = confusion_matrix(y_true, y_pred, labels=label)
    else:
        cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm


def get_trainable_parameter_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def L1Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(torch.abs(w))
    return err


def L2Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(w.pow(2))
    return err


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
       refer to: https://github.com/NVIDIA/DeepLearningExamples/blob/8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/ConvNets/image_classification/smoothing.py#L18
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class CustomLoss(nn.Module):
    def __init__(self, model, lambda_l1, lambda_l2):
        super(CustomLoss, self).__init__()
        self.model = model
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2

    def forward(self, output, target):
        cross_entropy_loss = F.cross_entropy(output, target)
        l1_loss = self.lambda_l1 * torch.sum(torch.abs(torch.cat([x.view(-1) for x in self.model.parameters()])))
        l2_loss = self.lambda_l2 * torch.sum(torch.pow(torch.cat([x.view(-1) for x in self.model.parameters()]), 2))
        total_loss = cross_entropy_loss + l1_loss + l2_loss
        return total_loss


class VisTech:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def heatmap_calculation(self, batchInput, sampleidx, state, radius=32):
        """
        Generate heatmap for the given sample after avgPool1 layer in the Baseline_7 model.
        """
        # Forward pass until avgPool1 layer
        with torch.no_grad():
            x = torch.unsqueeze(batchInput, 2)
            out1 = self.model.Depthwisconv1(x)
            out2 = self.model.Depthwisconv2(out1)
            out3 = self.model.Depthwisconv3(out2)
            out = torch.cat((out1, out2, out3), dim=2)
            weight = self.model.OneXOneConvWeight(out)
            out = out * weight
            out = self.model.BN1(out)
            out = self.model.activ(out)
            out = self.model.avgPool1(out)
            out = out.view(out.size(0), out.size(1), -1)
            print(out.size())
            sampleActiv = out[sampleidx]  # Activation after avgPool1 for the sample
            sampleActiv=torch.unsqueeze(sampleActiv,0)
            print(sampleActiv.size())
            # Processing through GCN layer
            adj = self.model.get_adj(sampleActiv)  # Get adjacency matrix
            sampleActiv = self.model.GCN(sampleActiv, adj)  # GCN output for the sample
            sampleActiv = sampleActiv.view(-1)  # Flatten the output for fc layer

        # Get the weights of the fully connected layer
        fc_weights = self.model.fc[1].weight.data  # Assuming the Linear layer is at index 1 in fc Sequential

        # Check if the size matches
        a=sampleActiv.shape
        b=fc_weights.shape

        # Generate Class Activation Map
        cam = torch.matmul(fc_weights[state], sampleActiv)
        cam = F.relu(cam)  # Apply ReLU to get the CAM

        # Reshape and Normalize CAM
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.cpu().numpy()

    def generate_heatmap(self, batchInput, sampleidx, subid, samplelabel, likelihood):
        """
        This function generates figures shown in the figure
        input:
           batchInput:          all the samples in a batch for classification
           sampleidx:           the index of the sample
           subid:               the ID of the subject
           samplelabel:         the ground truth label of the sample
           likelihood:          the likelihood of the sample to be classified into alert and drowsy state
        """

        if likelihood[0] > likelihood[1]:
            state = 0
        else:
            state = 1

        if samplelabel == 0:
            labelstr = 'alert'
        else:
            labelstr = 'drowsy'

        sampleInput = batchInput[sampleidx].cpu().detach().numpy().squeeze()
        sampleChannel = sampleInput.shape[0]
        sampleLength = sampleInput.shape[1]

        heatmap = self.heatmap_calculation(batchInput=batchInput, sampleidx=sampleidx, state=state)

        fig = plt.figure(figsize=(23, 6))

        gridlayout = gridspec.GridSpec(ncols=6, nrows=2, figure=fig, wspace=0.05, hspace=0.005)

        axs0 = fig.add_subplot(gridlayout[0:2, 0:2])
        axs1 = fig.add_subplot(gridlayout[0:2, 2:4])
        axs21 = fig.add_subplot(gridlayout[0, 4])
        axs22 = fig.add_subplot(gridlayout[0, 5])
        axs23 = fig.add_subplot(gridlayout[1, 4])
        axs24 = fig.add_subplot(gridlayout[1, 5])

        fig.suptitle('Subject:' + str(int(subid)) + '   ' + 'Label:' + labelstr + '   ' + '$P_{alert}=$' + str(
            round(likelihood[0], 2)) + '   $P_{drowsy}=$' + str(round(likelihood[1], 2)), y=1.02)
        thespan = np.percentile(sampleInput, 98)

        xx = np.arange(1, sampleLength + 1)
        for i in range(0, sampleChannel):
            y = sampleInput[i, :] + thespan * (sampleChannel - 1 - i)
            dydx = heatmap[i, :]

            points = np.array([xx, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(-1, 1)
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(dydx)
            lc.set_linewidth(2)
            axs0.add_collection(lc)

        yttics = np.zeros(sampleChannel)
        for gi in range(sampleChannel):
            yttics[gi] = gi * thespan

        axs0.set_ylim([-thespan, thespan * sampleChannel])
        axs0.set_xlim([0, sampleLength + 1])
        axs0.set_xticks([1, 100, 200, 300, 384])

        channelnames = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T3', 'C3', 'Cz',
                        'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2']

        inversechannelnames = []
        for i in range(sampleChannel):
            inversechannelnames.append(channelnames[sampleChannel - 1 - i])

        plt.sca(axs0)
        plt.yticks(yttics, inversechannelnames)

        deltapower = np.zeros(sampleChannel)
        thetapower = np.zeros(sampleChannel)
        alphapower = np.zeros(sampleChannel)
        betapower = np.zeros(sampleChannel)

        for kk in range(sampleChannel):
            psd, freqs = psd_array_multitaper(sampleInput[kk, :], 128, adaptive=True, normalization='full', verbose=0)
            freq_res = freqs[1] - freqs[0]

            totalpower = simps(psd, dx=freq_res)
            if totalpower < 0.00000001:
                deltapower[kk] = 0
                thetapower[kk] = 0
                alphapower[kk] = 0
                betapower[kk] = 0
            else:
                idx_band = np.logical_and(freqs >= 1, freqs <= 4)
                deltapower[kk] = simps(psd[idx_band], dx=freq_res) / totalpower
                idx_band = np.logical_and(freqs >= 4, freqs <= 8)
                thetapower[kk] = simps(psd[idx_band], dx=freq_res) / totalpower
                idx_band = np.logical_and(freqs >= 8, freqs <= 12)
                alphapower[kk] = simps(psd[idx_band], dx=freq_res) / totalpower
                idx_band = np.logical_and(freqs >= 12, freqs <= 30)
                betapower[kk] = simps(psd[idx_band], dx=freq_res) / totalpower

        axs21.set_title('Delta', y=-0.2)
        axs22.set_title('Theta', y=-0.2)
        axs23.set_title('Alpha', y=-0.2)
        axs24.set_title('Beta', y=-0.2)

        montage = 'standard_1020'
        sfreq = 128

        ch_names = channelnames

        info = mne.create_info(
            channelnames,
            ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg', \
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg', \
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg', \
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg', \
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg', \
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg'],
            sfreq=sfreq,
            montage=montage
        )

        topoHeatmap = np.mean(heatmap, axis=1)
        im, cn = mne.viz.plot_topomap(data=topoHeatmap, pos=info, vmin=-1, vmax=1, axes=axs1, names=ch_names,
                                      show_names=True, outlines='head', cmap='viridis', show=False)
        fig.colorbar(im, ax=axs1)

        mixpower = np.zeros((4, sampleChannel))
        mixpower[0, :] = deltapower
        mixpower[1, :] = thetapower
        mixpower[2, :] = alphapower
        mixpower[3, :] = betapower

        vmax = np.percentile(mixpower, 95)

        im, cn = mne.viz.plot_topomap(data=deltapower, pos=info, vmin=0, vmax=vmax, axes=axs21, names=ch_names,
                                      show_names=True, outlines='head', cmap='viridis', show=False)
        im, cn = mne.viz.plot_topomap(data=thetapower, pos=info, vmin=0, vmax=vmax, axes=axs22, names=ch_names,
                                      show_names=True, outlines='head', cmap='viridis', show=False)
        im, cn = mne.viz.plot_topomap(data=alphapower, pos=info, vmin=0, vmax=vmax, axes=axs23, names=ch_names,
                                      show_names=True, outlines='head', cmap='viridis', show=False)
        im, cn = mne.viz.plot_topomap(data=betapower, pos=info, vmin=0, vmax=vmax, axes=axs24, names=ch_names,
                                      show_names=True, outlines='head', cmap='viridis', show=False)

        fig.colorbar(im, ax=[axs21, axs22, axs23, axs24])

