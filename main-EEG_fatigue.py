# coding: utf-8
from cross_validation import *
from prepare_data_EEG_fatigue import *
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ######## Data ########
    parser.add_argument('--dataset', type=str, default='EEG-fatigue')
    parser.add_argument('--data-path', type=str, default='/data0/meya/code/cqq/dataset/EEG-fatigue')
    parser.add_argument('--subjects', type=int, default=12)
    parser.add_argument('--num-class', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--sampling-rate', type=int, default=128)
    parser.add_argument('--input-shape', type=tuple, default=(1,30,384))
    parser.add_argument('--segment', type=float, default=3)
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--data-format', type=str, default='eeg')
    parser.add_argument('---augmenData', type=bool, default=True)
    parser.add_argument('--augmenrate', type=float,default=5)
    parser.add_argument('--augmenelement', type=int, default=8)
    ######## Training Process ########Transformer
    parser.add_argument('--L1', type=float, default=1e-4)
    parser.add_argument('--L2', type=float, default=1e-2)
    parser.add_argument('--fold', type=int, default=10)
    parser.add_argument('--random-seed', type=int, default=2023)
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--patient', type=int, default=20)
    parser.add_argument('--max-epoch-cmb', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--save-data', default='/data0/meya/code/cqq/LGGNet/data')
    parser.add_argument('--result-path', default='/data0/meya/code/cqq/LGGNet/result')
    parser.add_argument('--save-path', default='./save_EEG-fatigue/')
    parser.add_argument('--load-path', default='./save_EEG-fatigue/max-acc.pth')
    parser.add_argument('--load-path-final', default='./save_EEG-fatigue/final_model.pth')
    parser.add_argument('--save-model', type=bool, default=True)
    ######## Model Parameters ########
    parser.add_argument('--train', type=str, default='independent')  
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--model', type=str, default='NHGCN')
    parser.add_argument('--pool', type=int, default=64)
    parser.add_argument('--pool-step', type=int, default=1)
    parser.add_argument('--gpu',type=str, default='0')
    parser.add_argument('--outGraph_1', type=int, default=32)
    parser.add_argument('--PartitionData', type=bool, default=True)
    parser.add_argument('--PartitionRate', type=float, default=0.5)

    args = parser.parse_args()
    sub_to_run = np.arange(1,args.subjects+1)

    pd = PrepareData_EEG_fatigue(args)
    pd.run(sub_to_run)


    cv = CrossValidation(args)
    seed_all(args.random_seed)
    if args.train=='dependent':
        cv.dependent_n_fold_CV(sub_to_run)
    elif args.train=='independent':
        cv.independent_n_fold_CV(sub_to_run)
