# This is the processing script of fatigue(Dataset 1) dataset
from train_model import *
import scipy.io as sio
import numpy as np
import os
import mne
from sklearn.model_selection import StratifiedShuffleSplit
import pykalman

class PrepareData_fatigue:
    def __init__(self, args):
        # init all the parameters here
        # arg contains parameter settings
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        self.data_path = args.data_path

    def run(self, subject_list):
        """
        Parameters
        ----------
        subject_list: the subjects need to be processed
        -------
        The processed data will be saved './data_<data_format>_<dataset>/sub1.hdf'   处理后的数据将保存到指定位置
        """
        for sub in subject_list:
            data_, label_ = self.load_data_per_subject(sub)
            if self.args.augmenData==True:
                augmen=True
                augmentdata_, augmentlabel_=self.augmentData(data_,label_)
                print('augmentdata_:' + str(augmentdata_.shape) + ' augmentlabel_:' + str(augmentlabel_.shape))
                self.save(augmentdata_, augmentlabel_, sub,augmen)
            if self.args.PartitionData==True:
                self.dis_data(data_,label_,sub)
            print('Data and label prepared!')
            print('----------------------')
            augmen=False
            self.save(data_, label_, sub,augmen)

    def load_data_per_subject(self, sub):
        data='unbalanced_dataset.mat'
        data_path=os.path.join(self.args.data_path, data)
        data = sio.loadmat(data_path)
        xdata = np.array(data['EEGsample'])
        label = np.array(data['substate'])
        subIdx = np.array(data['subindex'])
        label.astype(int)
        subIdx.astype(int)
        samplenum = label.shape[0]
        ydata = np.zeros(samplenum, dtype=np.longlong)
        sub_id = np.zeros(samplenum, dtype=int)
        for i in range(samplenum):
            ydata[i] = label[i]
        for i in range(samplenum):
            sub_id[i] = subIdx[i]
        sub_indx = np.where(sub_id == sub)[0]
        sub_data = xdata[sub_indx]
        sub_label = ydata[sub_indx]
        print('data:' + str(sub_data.shape) + ' label:' + str(sub_label.shape))
        return sub_data, sub_label

    def augmentData(self,data,label):
        aug_data = []
        aug_label = []
        for cls4aug in range(self.args.num_class):
            cls_idx = np.where(label == cls4aug)  # 每一类进行数据增强
            tmp_data = data[cls_idx]
            n = int(tmp_data.shape[0] * self.args.augmenrate)
            tmp_aug_data = np.zeros((n, tmp_data.shape[1], tmp_data.shape[2]))  # 不是很理解为什么只需要batch_size/4的数量
            for ri in range(n):
                for rj in range(self.args.augmenelement):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], self.args.augmenelement)  # 0-94中随机取8个数，也就是94个样本中随机取8个样本
                    tmp_aug_data[ri, :, rj * (tmp_data.shape[2] // self.args.augmenelement):(rj + 1) * (tmp_data.shape[2] // self.args.augmenelement)] = tmp_data[rand_idx[rj], :,rj * (tmp_data.shape[2] // self.args.augmenelement):(rj + 1) * (tmp_data.shape[2] // self.args.augmenelement)]
            aug_data.append(tmp_aug_data)
            aug_label.append(np.full(n, cls4aug))
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        return aug_data, aug_label

    def dis_data(self,data_, label_,sub):

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.args.PartitionRate,
                                          random_state=self.args.random_seed)

        for train_index, test_index in splitter.split(data_, label_):
            data_TF, data_test = data_[train_index], data_[test_index]
            label_TF, label_test = label_[train_index], label_[test_index]

        save_path = self.args.save_data
        data_type = 'data_{}_{}'.format(self.args.data_format, self.args.dataset)

        name = 'sub' + str(sub) + '.hdf'
        save_path = osp.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        #Transfer
        val_save_path = osp.join(save_path, 'Transfer_{}'.format(self.args.PartitionRate))
        if not os.path.exists(val_save_path):
            os.makedirs(val_save_path)
        val_save_path = osp.join(val_save_path, name)
        dataset = h5py.File(val_save_path, 'w')
        dataset['data'] = data_TF
        dataset['label'] = label_TF
        dataset.close()

        #test
        test_save_path = osp.join(save_path, 'Test_{}'.format(self.args.PartitionRate))
        if not os.path.exists(test_save_path):
            os.makedirs(test_save_path)
        test_save_path = osp.join(test_save_path, name)
        dataset = h5py.File(test_save_path, 'w')
        dataset['data'] = data_test
        dataset['label'] = label_test
        dataset.close()


    def save(self, data, label, sub,augmen):
        """
        This function save the processed data into target folder
        Parameters
        ----------
        data: the processed data
        label: the corresponding label
        sub: the subject ID

        Returns
        -------
        None
        """
        save_path = self.args.save_data
        data_type = 'data_{}_{}'.format(self.args.data_format, self.args.dataset)
        if self.args.augmenData == True:
            augmenData_type = 'augmenData_{}_rate{}'.format(self.args.augmenelement, self.args.augmenrate)
        if augmen==False:
            path = osp.join(save_path, data_type)
        elif augmen==True:
            path = osp.join(save_path,data_type,augmenData_type)
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            pass
        name = 'sub' + str(sub) + '.hdf'
        save_path=osp.join(path,name)
        dataset = h5py.File(save_path, 'w')
        dataset['data'] = data
        dataset['label'] = label
        dataset.close()
