from utils import *
import torch.nn as nn
from sklearn.metrics import recall_score,f1_score,precision_score
import numpy as np
import os.path as osp
import multiprocessing as mp
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import warnings
import torch
import os
import matplotlib.pyplot as plt
import seaborn
from matplotlib import gridspec
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore",category=UserWarning)
CUDA = torch.cuda.is_available()


def train_one_epoch(args, data_loader, net, loss_fn, optimizer):
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        if CUDA:
            device = torch.device("cuda:{}".format(args.gpu))
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        out = net(x_batch)
        loss = loss_fn(out, y_batch)
        _, pred = torch.max(out, 1)
        tl.add(loss)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return tl.item(), pred_train, act_train


def predict(args,data_loader, net, loss_fn):
    net.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(data_loader):
            if CUDA:
                device = torch.device("cuda:{}".format(args.gpu))
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            out = net(x_batch)
            loss = loss_fn(out, y_batch)
            _, pred = torch.max(out, 1)
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
    return vl.item(), pred_val, act_val


def set_up(args):
    set_gpu(args.gpu)
    ensure_path(args.save_path)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True


def train(args, data_train, label_train, data_val, label_val, subject, fold):

    seed_all(args.random_seed)
    save_name = '_sub' + str(subject) + '_fold' + str(fold)
    set_up(args)

    train_loader = get_dataloader(data_train, label_train, args.batch_size)

    val_loader = get_dataloader(data_val, label_val, args.batch_size)
    model = get_model(args,subject)
    model_map=get_model(args,subject)
    loss_fn = CustomLoss(model, args.L1, args.L2)
    if CUDA:
        device = torch.device("cuda:{}".format(args.gpu))
        model = model.to(device)
        model_map= model_map.to(device)
        # data = torch.tensor(torch.randn(256, 30, 384), device=device)

    # #plot_model of keras
    # model_graph = draw_graph(model_map, input_size=(64, 30, 384), expand_nested=True, save_graph=True, filename="/media/disk1/meya_code/cqq/LGGNet/Baseline_7_1207",
    #                   directory=".")
    # model_graph.visual_graph
    summary(model_map, (30, 384))

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


    def save_model(name):
        previous_model = osp.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(args.save_path, '{}.pth'.format(name)))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['F1'] = 0.0

    timer = Timer()
    patient = args.patient
    counter = 0

    for epoch in range(1, args.max_epoch + 1):
        loss_train, pred_train, act_train = train_one_epoch(
            args=args, data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer)

        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train, classes=args.num_class)
        print('epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'.format(epoch, loss_train, acc_train, f1_train))

        loss_val, pred_val, act_val = predict(args=args, data_loader=val_loader, net=model, loss_fn=loss_fn)
        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val, classes=args.num_class)
        print('epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'.format(epoch, loss_val, acc_val, f1_val))


        if acc_val >= trlog['max_acc']:
            trlog['max_acc'] = acc_val
            trlog['F1'] = f1_val
            save_model('candidate')
            counter = 0
        else:
            counter += 1
            if counter >= patient:
                print('early stopping')
                break

        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)

        print('ETA:{}/{} SUB:{} FOLD:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                                 subject, fold))
    return trlog['max_acc'], trlog['F1']


def test(args, data, label, subject, fold,sub):
    set_up(args)
    seed_all(args.random_seed)
    test_loader = get_dataloader(data, label, args.batch_size)

    model = get_model(args,sub)

    if args.train=='dependent':
        checkpoint = torch.load(args.load_path,map_location='cpu')  # args.load_path ./save_fatigue_2HZ_DE_myLDS/max-acc.pth
        model.load_state_dict(checkpoint)
    else:
        checkpoint = torch.load(args.load_path_final,map_location='cpu')  # args.load_path ./save_fatigue_2HZ_DE_myLDS/max-acc.pth
        model.load_state_dict(checkpoint)

    loss_fn = CustomLoss(model, args.L1, args.L2)


    model.load_state_dict(checkpoint)

    # Now move the model to the GPU if available
    if CUDA:
        device = torch.device("cuda:{}".format(args.gpu))
        model = model.to(device)

    loss, pred, act = predict(
        args=args,data_loader=test_loader, net=model, loss_fn=loss_fn
    )
    if args.dataset=='SEED-VIG' and (subject==8 or subject==15):
        num_class=2
    else:
        num_class=args.num_class
    acc, f1, cm = get_metrics(y_pred=pred, y_true=act,classes=num_class)
    save_xlsx(args,sub,fold,acc, f1, cm,pred,act)
    print('>>> Test:  loss={:.4f} acc={:.4f} f1={:.4f}'.format(loss, acc, f1))
    return acc, pred, act


def combine_train(args, data, label, subject, fold, target_acc):

    save_name = '_sub' + str(subject) + '_fold' + str(fold)  #'_sub1_fold0'
    set_up(args)
    seed_all(args.random_seed)
    train_loader = get_dataloader(data, label, args.batch_size)
    model = get_model(args, subject)

    checkpoint = torch.load(args.load_path,map_location='cpu')  # args.load_path ./save_fatigue_2HZ_DE_myLDS/max-acc.pth
    model.load_state_dict(checkpoint)

    # Now move the model to the GPU if available
    if CUDA:
        device = torch.device("cuda:{}".format(args.gpu))
        model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate*1e-1)

    loss_fn = CustomLoss(model, args.L1, args.L2)

    def save_model(name):
        previous_model = osp.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(args.save_path, '{}.pth'.format(name)))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch_cmb + 1):
        loss, pred, act = train_one_epoch(
            args=args,data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer
        )
        acc, f1, _ = get_metrics(y_pred=pred, y_true=act,classes=args.num_class)
        print('Stage 2 : epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss, acc, f1))

        if acc >= target_acc or epoch == args.max_epoch_cmb:
            print('early stopping!')
            save_model('final_model')
            break

        trlog['train_loss'].append(loss)
        trlog['train_acc'].append(acc)

        print('ETA:{}/{} SUB:{} TRIAL:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                                 subject, fold))



def save_xlsx(args,sub,fold,acc, f1, cm,y_pred,y_true):

    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary')
    specificity=TN/(TN+FP)


    result_path = args.result_path
    data_type = 'data_{}_{}'.format(args.data_format, args.dataset)
    if args.augmenData == True:
        data_type = data_type + '_augmenData_{}_rate{}'.format(args.augmenelement,args.augmenrate)
    data_type =data_type+ '_{}flod'.format(args.fold)

    if args.train=='dependent':
        data_type=data_type+'_dependent'
    elif args.train=='independent':
        data_type = data_type + '_independent_Trans{}'.format(args.PartitionRate)
    data_type=data_type+'_bs{}'.format(args.batch_size)
    data_type = data_type + '_avepool{}_{}_outGraph{}'.format(args.pool, args.pool_step, args.outGraph_1)
    xlsxpath = osp.join(result_path + '/' + data_type + "_{}_results.xlsx".format(args.model,args.dropout))
    data = [sub, fold, TP, FP, TN, FN, acc, precision, sensitivity, specificity, f1]
    lock = mp.Lock()
    with lock:
        try:
            with pd.ExcelFile(xlsxpath) as xlsx:
                df = pd.read_excel(xlsx)
        except FileNotFoundError:
            df = pd.DataFrame(
                columns=['sub', 'fold', 'TP', 'FP', 'TN', 'FN', 'acc', 'precision', 'sensitivity',
                         'specificity', 'f1_score'])
        # 将数据追加到DataFrame中
        new_row = pd.Series(data, index=df.columns)
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
        # 使用with语句将DataFrame写入Excel文件
        with pd.ExcelWriter(xlsxpath) as writer:
            df.to_excel(writer, index=False)