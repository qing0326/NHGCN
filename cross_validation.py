# coding: utf-8
from train_model import *
from utils import Averager
from sklearn.model_selection import StratifiedKFold
import os

ROOT = os.getcwd()

class CrossValidation:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = None

    def load_other_subject(self, sub,augmen):
        """
        load data for sub
        :param sub: which subject's data to load
        :return: data and label
        """
        save_path = self.args.save_data
        data_type = 'data_{}_{}'.format(self.args.data_format, self.args.dataset)
        if self.args.augmenData == True:
            augmenData_type = 'augmenData_{}_rate{}'.format(self.args.augmenelement, self.args.augmenrate)
        if augmen == False:
            save_path = osp.join(save_path, data_type)
        elif augmen == True:
            save_path = osp.join(save_path, data_type, augmenData_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        other_sub = np.arange(1, self.args.subjects + 1)
        other_sub = np.delete(other_sub, np.where(other_sub == sub ))
        train_data=[]
        train_label=[]
        for subject in other_sub:
            name = 'sub' + str(subject) + '.hdf'
            path = osp.join(save_path, name)
            with h5py.File(path, 'r') as dataset:
                data = np.array(dataset['data'])
                label = np.array(dataset['label'])
                train_data.append(data)
                train_label.append(label)
        train_data=np.concatenate(train_data)
        train_label=np.concatenate(train_label)
        print('>>> Data:{} Label:{}'.format(train_data.shape, train_label.shape))
        return train_data, train_label

    def load_per_subject(self, sub,augmen):
        save_path = self.args.save_data
        data_type = 'data_{}_{}'.format(self.args.data_format, self.args.dataset)
        if self.args.augmenData == True:
            augmenData_type = 'augmenData_{}_rate{}'.format(self.args.augmenelement, self.args.augmenrate)
        if augmen == False:
            save_path = osp.join(save_path, data_type)
        elif augmen == True:
            save_path = osp.join(save_path, data_type, augmenData_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        name = 'sub' + str(sub) + '.hdf'
        if self.args.train=='independent':
            test_save_path = osp.join(save_path, 'Test_{}'.format(self.args.PartitionRate))
            test_save_path = osp.join(test_save_path, name)
            with h5py.File(test_save_path, 'r') as dataset:
                data = np.array(dataset['data'])
                label = np.array(dataset['label'])
        else:
            path = osp.join(save_path, name)
            with h5py.File(path, 'r') as dataset:
                data = np.array(dataset['data'])
                label = np.array(dataset['label'])
        return data, label

    def prepare_data(self,sub,idx_train, idx_test, data, label):
        data_train = data[idx_train]
        label_train = label[idx_train]
        data_test = data[idx_test]
        label_test = label[idx_test]
        return data_train, label_train, data_test, label_test

    #dependent
    def dependent_n_fold_CV(self, subject):
        tta = []  # total test accuracy
        tva = []  # total validation accuracy
        ttf = []  # total test f1
        tvf = []  # total validation f1
        fold=self.args.fold
        for sub in subject:
            augmen = False
            data, label = self.load_per_subject(sub,augmen)
            va_val = Averager()
            vf_val = Averager()
            preds, acts = [], []
            skf = StratifiedKFold(n_splits=fold, shuffle=True)
            for idx_fold, (idx_train, idx_test) in enumerate(skf.split(data, label)):
                print('Outer loop: {}-fold-CV Fold:{}'.format(fold, idx_fold))
                data_train, label_train, data_test, label_test = self.prepare_data(
                    sub=sub,idx_train=idx_train, idx_test=idx_test, data=data, label=label)
                print('>>> trainData:{} trainLabel:{}'.format(data_train.shape, label_train.shape))
                print('>>> TestData:{} TestLabel:{}'.format(data_test.shape, label_test.shape))
                acc_val, f1_val= self.first_stage_dependent(data=data_train, label=label_train,
                                                   subject=sub, fold=idx_fold)

                data_test = torch.from_numpy(data_test).float()
                label_test = torch.from_numpy(label_test).long()
                acc_test, pred, act = test(args=self.args, data=data_test, label=label_test,
                                           subject=sub, fold=idx_fold,sub=sub)
                va_val.add(acc_val)
                vf_val.add(f1_val)
                preds.extend(pred)
                acts.extend(act)

            tva.append(va_val.item())
            tvf.append(vf_val.item())
            acc, f1, _ = get_metrics(y_pred=preds, y_true=acts,classes=self.args.num_class)
            tta.append(acc)
            ttf.append(f1)

        # prepare final report
        tta = np.array(tta)
        ttf = np.array(ttf)
        tva = np.array(tva)
        tvf = np.array(tvf)
        mACC = np.mean(tta)
        mF1 = np.mean(ttf)
        std = np.std(tta)
        mACC_val = np.mean(tva)
        std_val = np.std(tva)
        mF1_val = np.mean(tvf)

        print('Final: test mean ACC:{} std:{}'.format(mACC, std))
        print('Final: val mean ACC:{} std:{}'.format(mACC_val, std_val))
        print('Final: val mean F1:{}'.format(mF1_val))

    #independent
    def independent_n_fold_CV(self,subject=[0], fold=10, shuffle=True):
        """
        this function achieves n-fold cross-validation
        :param subject: how many subject to load
        :param fold: how many fold
        """
        # Train and evaluate the model subject by subject
        # Train and evaluate the model subject by subject
        tta = []  # total test accuracy
        tva = []  # total validation accuracy
        ttf = []  # total test f1
        tvf = []  # total validation f1
        fold=self.args.fold
        for sub in subject:
            data,label=self.load_other_subject(sub,False)
            if self.args.augmenData==True:
                augmendata,augmenabel=self.load_other_subject(sub,True)
                data=np.concatenate((data, augmendata), axis=0)
                label=np.concatenate((label, augmenabel), axis=0)
            data_test,label_test=self.load_per_subject(sub, False)
            data_test = torch.from_numpy(data_test).float()
            label_test = torch.from_numpy(label_test).long()
            va_val = Averager()
            vf_val = Averager()
            preds, acts = [], []
            skf = StratifiedKFold(n_splits=fold, shuffle=shuffle)
            for idx_fold, (idx_train, _) in enumerate(skf.split(data, label)):
                print('Outer loop: {}-fold-CV Fold:{}'.format(fold, idx_fold))
                data_train = data[idx_train]
                label_train = label[idx_train]
                acc_val, f1_val = self.first_stage_independent(data=data_train, label=label_train,
                                                   subject=sub, fold=idx_fold)
                save_path = osp.join(self.args.save_data, 'data_eeg_{}'.format(self.args.dataset))
                name = 'sub' + str(sub) + '.hdf'
                val_save_path = osp.join(save_path, 'Transfer_{}'.format(self.args.PartitionRate))
                val_save_path = osp.join(val_save_path, name)
                with h5py.File(val_save_path, 'r') as dataset:
                    data_TL = np.array(dataset['data'])
                    label_TL = np.array(dataset['label'])
                data_TL = torch.from_numpy(data_TL).float()
                label_TL = torch.from_numpy(label_TL).long()
                combine_train(args=self.args,
                              data=data_TL, label=label_TL,
                              subject=sub, fold=idx_fold, target_acc=1)
                acc_test, pred, act = test(args=self.args, data=data_test, label=label_test,
                                           subject=sub, fold=idx_fold,sub=sub)
                va_val.add(acc_val)
                vf_val.add(f1_val)
                preds.extend(pred)
                acts.extend(act)

            tva.append(va_val.item())
            tvf.append(vf_val.item())
            acc, f1, _ = get_metrics(y_pred=preds, y_true=acts,classes=self.args.num_class)
            tta.append(acc)
            ttf.append(f1)

        # prepare final report
        tta = np.array(tta)
        ttf = np.array(ttf)
        tva = np.array(tva)
        tvf = np.array(tvf)
        mACC = np.mean(tta)
        mF1 = np.mean(ttf)
        std = np.std(tta)
        mACC_val = np.mean(tva)
        std_val = np.std(tva)
        mF1_val = np.mean(tvf)

        print('Final: test mean ACC:{} std:{}'.format(mACC, std))
        print('Final: val mean ACC:{} std:{}'.format(mACC_val, std_val))
        print('Final: val mean F1:{}'.format(mF1_val))

    def first_stage_dependent(self, data, label, subject, fold):
        has_zero = (data == 0).any()
        skf = StratifiedKFold(n_splits=3, shuffle=True)
        va = Averager()
        vf = Averager()
        va_item = []
        maxAcc = 0.0
        for i, (idx_train, idx_val) in enumerate(skf.split(data, label)):
            print('Inner 3-fold-CV Fold:{}'.format(i))
            data_train,label_train=data[idx_train], label[idx_train]
            data_val, label_val = data[idx_val], label[idx_val]
            if self.args.augmenData == True:
                augmen_data_train, augmen_label_train = self.augmentData(data_train, label_train)
                data_train = np.concatenate((augmen_data_train, data_train), axis=0)
                label_train = np.concatenate((augmen_label_train, label_train), axis=0)

            perm_train = np.random.permutation(len(label_train))
            data_train = data_train[perm_train]
            label_train = label_train[perm_train]
            data_train = torch.from_numpy(data_train).float()
            label_train = torch.from_numpy(label_train).long()

            perm_val = np.random.permutation(len(label_val))
            data_val = data_val[perm_val]
            label_val = label_val[perm_val]
            data_val = torch.from_numpy(data_val).float()
            label_val = torch.from_numpy(label_val).long()

            acc_val, F1_val = train(args=self.args,
                                    data_train=data_train,
                                    label_train=label_train,
                                    data_val=data_val,
                                    label_val=label_val,
                                    subject=subject,
                                    fold=fold)
            va.add(acc_val)
            vf.add(F1_val)
            va_item.append(acc_val)
            if acc_val >= maxAcc:
                maxAcc = acc_val
                # choose the model with higher val acc as the model to second stage
                old_name = osp.join(self.args.save_path, 'candidate.pth')  # './save_fatigue_2HZ_DE_myLDS/candidate.pth'
                new_name = osp.join(self.args.save_path, 'max-acc.pth')  # './save_fatigue_2HZ_DE_myLDS/max-acc.pth'

                try:
                    # 如果目标文件已存在，先删除它
                    if os.path.exists(new_name):
                        os.remove(new_name)

                    # 检查源文件是否存在，然后重命名
                    if os.path.exists(old_name):
                        os.rename(old_name, new_name)
                    else:
                        print(f"Warning: Source file {old_name} does not exist!")

                except Exception as e:
                    print(f"Error occurred while handling files: {e}")
                print('New max ACC model saved, with the val ACC being:{}'.format(acc_val))

        mAcc = va.item()
        mF1 = vf.item()
        return mAcc, mF1


    def first_stage_independent(self, data, label, subject, fold):
        has_zero = (data == 0).any()
        skf = StratifiedKFold(n_splits=3, shuffle=True)
        va = Averager()
        vf = Averager()
        va_item = []
        maxAcc = 0.0

        for i, (idx_train, idx_val) in enumerate(skf.split(data, label)):
            print('Inner 3-fold-CV Fold:{}'.format(i))
            data_train,label_train=data[idx_train], label[idx_train]
            data_val, label_val = data[idx_val], label[idx_val]

            perm_train = np.random.permutation(len(label_train))
            data_train = data_train[perm_train]
            label_train = label_train[perm_train]

            perm_val = np.random.permutation(len(label_val))
            data_val = data_val[perm_val]
            label_val = label_val[perm_val]

            data_train = torch.from_numpy(data_train).float()
            label_train = torch.from_numpy(label_train).long()
            data_val = torch.from_numpy(data_val).float()
            label_val = torch.from_numpy(label_val).long()

            acc_val, F1_val = train(args=self.args,
                                    data_train=data_train,
                                    label_train=label_train,
                                    data_val=data_val,
                                    label_val=label_val,
                                    subject=subject,
                                    fold=fold)
            va.add(acc_val)
            vf.add(F1_val)
            va_item.append(acc_val)
            if acc_val >= maxAcc:
                maxAcc = acc_val
                # choose the model with higher val acc as the model to second stage
                old_name = osp.join(self.args.save_path, 'candidate.pth')  # './save_fatigue_2HZ_DE_myLDS/candidate.pth'
                new_name = osp.join(self.args.save_path, 'max-acc.pth')  # './save_fatigue_2HZ_DE_myLDS/max-acc.pth'

                try:
                    # 如果目标文件已存在，先删除它
                    if os.path.exists(new_name):
                        os.remove(new_name)

                    # 检查源文件是否存在，然后重命名
                    if os.path.exists(old_name):
                        os.rename(old_name, new_name)
                    else:
                        print(f"Warning: Source file {old_name} does not exist!")

                except Exception as e:
                    print(f"Error occurred while handling files: {e}")
                print('New max ACC model saved, with the val ACC being:{}'.format(acc_val))

        mAcc = va.item()
        mF1 = vf.item()
        return mAcc, mF1

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
