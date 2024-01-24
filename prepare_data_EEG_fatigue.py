# This is the processing script of DEAP dataset
from train_model import *
from sklearn.model_selection import StratifiedShuffleSplit
import mne


class PrepareData_EEG_fatigue:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = None

    def run(self, subject_list, split=True):
        """
        Parameters
        ----------
        subject_list: the subjects need to be processed
        split: (bool) whether to split one trial's data into shorter segment


        Returns
        -------
        The processed data will be saved './data_<data_format>_<dataset>/sub1.hdf'
        """
        for sub in subject_list:
            Normal_data, Fatigue_data = self.load_data_per_subject(sub)
            if split:
                data_, label_ = self.split(
                    Normal_data=Normal_data, Fatigue_data=Fatigue_data, segment_length=self.args.segment,
                    overlap=self.args.overlap, sampling_rate=self.args.sampling_rate)
            if self.args.augmenData == True:
                augmen = True
                augmentdata_, augmentlabel_ = self.augmentData(data_, label_)
                print('augmentdata_:' + str(augmentdata_.shape) + ' augmentlabel_:' + str(augmentlabel_.shape))
                self.save(augmentdata_, augmentlabel_, sub, augmen)
            if self.args.PartitionData==True:
                self.dis_data(data_,label_,sub)
            print('Data and label prepared!')
            print('data:' + str(data_.shape) + ' label:' + str(label_.shape))
            print('----------------------')
            augmen = False
            self.save(data_, label_, sub,augmen)

    def load_data_per_subject(self, sub):
        Normal_file_path = '{}/{}/Normal state.cnt'.format(self.args.data_path,sub)
        Normal_raw = mne.io.read_raw_cnt(Normal_file_path, preload=True)
        rename_dict = {
            'FP1': 'Fp1',
            'FP2': 'Fp2',
            'FZ' : 'Fz',
            'FCZ': 'FCz',
            'CZ' : 'Cz',
            'CPZ':'CPz',
            'PZ' : 'Pz',
            'OZ' : 'Oz'
        }
        Normal_raw.rename_channels(rename_dict)

        #选择通道
        channels_to_keep =['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8',
         'T3', 'C3', 'Cz', 'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'T5', 'P3', 'Pz', 'P4', 'T6',
         'O1', 'Oz', 'O2']
        Normal_raw.pick_channels(channels_to_keep)
        #滤波
        Normal_raw.notch_filter(50)
        Normal_raw.filter(0.15, 45)

        #下采样
        Normal_raw.resample(self.args.sampling_rate)
        #裁剪
        sampling_rate = Normal_raw.info['sfreq']
        five_minutes_samples = 5 * 60 * sampling_rate
        Normal_data=Normal_raw.get_data()
        start_point_Normal = int(Normal_data.shape[1] - five_minutes_samples)
        last_five_Normal_data = Normal_data[:, start_point_Normal:]

        #疲劳数据
        Fatigue_file_path = '{}/{}/Fatigue state.cnt'.format(self.args.data_path, sub)
        Fatigue_raw = mne.io.read_raw_cnt(Fatigue_file_path, preload=True)
        Fatigue_raw.rename_channels(rename_dict)

        # 选择通道
        Fatigue_raw.pick_channels(channels_to_keep)

        # 滤波
        Fatigue_raw.notch_filter(50)
        Fatigue_raw.filter(0.15, 45)

        # 降采样
        Fatigue_raw.resample(self.args.sampling_rate)
        # 裁剪
        sampling_rate = Fatigue_raw.info['sfreq']
        five_minutes_samples = 5 * 60 * sampling_rate
        Fatigue_data = Fatigue_raw.get_data()
        start_point_Fatigue = int(Fatigue_data.shape[1] - five_minutes_samples)
        last_five_Fatigue_data = Fatigue_data[:, start_point_Fatigue:]

        last_five_Normal_data = last_five_Normal_data * 1e6
        last_five_Fatigue_data=last_five_Fatigue_data* 1e6
        return last_five_Normal_data, last_five_Fatigue_data



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

    def split(self, Normal_data, Fatigue_data, segment_length, overlap, sampling_rate):
        segment_samples = segment_length * sampling_rate
        step_size = segment_samples * (1 - overlap)
        num_segments = int((Normal_data.shape[1] - segment_samples) / step_size) + 1
        X_Normal = np.zeros((num_segments, Normal_data.shape[0], segment_samples))
        for i in range(num_segments):
            start_idx = i * step_size
            end_idx = start_idx + segment_samples
            X_Normal[i, :, :] = Normal_data[:, start_idx:end_idx]
        Y_Normal = np.zeros(num_segments)

        X_Fatigue = np.zeros((num_segments, Fatigue_data.shape[0], segment_samples))
        for i in range(num_segments):
            start_idx = i * step_size
            end_idx = start_idx + segment_samples
            X_Fatigue[i, :, :] = Fatigue_data[:, start_idx:end_idx]
        Y_Fatigue = np.ones(num_segments)
        data = np.vstack((X_Normal, X_Fatigue))
        label = np.concatenate((Y_Normal, Y_Fatigue))

        assert len(data) == len(label)
        return data, label

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

    def dis_data(self, data_, label_, sub):

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.args.PartitionRate,
                                          random_state=self.args.random_seed)

        for train_index, test_index in splitter.split(data_, label_):
            data_val, data_test = data_[train_index], data_[test_index]
            label_val, label_test = label_[train_index], label_[test_index]

        save_path = self.args.save_data
        data_type = 'data_{}_{}'.format(self.args.data_format, self.args.dataset)

        name = 'sub' + str(sub) + '.hdf'
        save_path = osp.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        # val
        val_save_path = osp.join(save_path, 'Transfer_{}'.format(self.args.PartitionRate))
        if not os.path.exists(val_save_path):
            os.makedirs(val_save_path)
        val_save_path = osp.join(val_save_path, name)
        dataset = h5py.File(val_save_path, 'w')
        dataset['data'] = data_val
        dataset['label'] = label_val
        dataset.close()

        # test
        test_save_path = osp.join(save_path, 'Test_{}'.format(self.args.PartitionRate))
        if not os.path.exists(test_save_path):
            os.makedirs(test_save_path)
        test_save_path = osp.join(test_save_path, name)
        dataset = h5py.File(test_save_path, 'w')
        dataset['data'] = data_test
        dataset['label'] = label_test
        dataset.close()
