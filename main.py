#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   
'''
import sys
# sys.path.append(r"f:\20220330PredictExp\0Code\3TimeSformer")
sys.path.append(r"C:\Users\DELL\Desktop\code20140117\transformer")
#sys.path.pop( )
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import argparse
from torch.autograd import Variable
from transformer_raw import SpaceTimeTransformer,TimesFormer_2input
from config import configs
import pickle
from datasets import read_data,ImageDataset_array
import h5py,timeit,time
from math import *
import csv




# from utils import balance_samples

# TIMESTAMP = "2022-03-31"




def train():
    '''
    main function to run the training
    '''
    print(configs.__dict__)
    net = TimesFormer_2input(configs).to(configs.device)
    
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=15, verbose=True, delta = 1e-6)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0
        
    lossfunction = nn.MSELoss().cuda() #损失函数改这里 #nn.MSELoss #nn.BCEWithLogitsLoss() #CrossEntropyLoss
    optimizer = optim.Adam(net.parameters(), lr=configs.lr) #优化方案是Adam
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=10,
                                                      verbose=True)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # mini_val_loss = np.inf
    for epoch in range(cur_epoch, configs.num_epochs + 1):
        ###################
        # train the model #
        ###################
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        #for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
        for i, imgs in enumerate(t):

            inputs1 = Variable(imgs["Ix1"]).to(device)  # B,S,C,H,W
            inputs2 = Variable(imgs["Ix2"]).to(device)  # B,S,C,H,W
            label = Variable(imgs["Iy"]).to(device)  # B,S,C,H,W
            
            tgt = label.repeat(1, 1, 10, 1, 1)
            inputs2 = torch.squeeze(inputs2,1) # B,C,H,W
            
            optimizer.zero_grad()
            net.train()
            label_reshape = label.squeeze() # 调整label的shape，使其与model output一致
            label_reshape = label_reshape[:,:,:,None] # 调整label的shape，使其与model output一致
            pred = net(inputs1=inputs1,inputs2=inputs2,tgt=tgt, train=True) # B,S,C,H,W
            loss = lossfunction(pred, label_reshape)
            loss_aver = loss.item() / configs.batch_size
            train_losses.append(loss_aver)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()
            t.set_postfix({
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
        tb.add_scalar('TrainLoss', loss_aver, epoch)
        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, imgs in enumerate(t):
                if i == 3000:
                    break
                inputs1 = Variable(imgs["Ix1"]).to(device)  # B,S,C,H,W
                inputs2 = Variable(imgs["Ix2"]).to(device)  # B,S,C,H,W
                label = Variable(imgs["Iy"]).to(device)  # B,S,C,H,W
                
                inputs2 = torch.squeeze(inputs2,1)
                label_reshape = label.squeeze() # 调整label的shape，使其与model output一致
                label_reshape = label_reshape[:,:,:,None] # 调整label的shape，使其与model output一致
                pred = net(inputs1=inputs1,inputs2=inputs2,tgt=None, train=False) # B,S,C,H,W
                loss = lossfunction(pred, label_reshape)
                loss_aver = loss.item() / configs.batch_size_test
                # record validation loss
                valid_losses.append(loss_aver)
                #print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })

        tb.add_scalar('ValidLoss', loss_aver, epoch)
        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(configs.num_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{configs.num_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open(r"d:\DL\para-test\08-5-20241029\avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open(r"d:\DL\para-test\08-5-20241029\avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)
def read_samples0803(rootdir): 
    s_y = []
    s_x1 = []  #动态变量
    s_x2 = []  #静态变量
    num=0
    for dirpath, filename, filenames in os.walk(rootdir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.h5':
                num+= 1
                if num%100 == 0:
                    print(num)
                # print (filename)                
                s_name = os.path.join(dirpath, filename)
                f = h5py.File(s_name,"r")
                for group in f.keys():
                    # print (group)
                    group_read = f[group]
                    data = np.array(group_read)
                    #print(data.shape)
                    if group =='y':
                        for i in range(0,4):
                            for j in range(0,4):
                                s_y.append(data[i*8:i*8+8,j*8:j*8+8])
                                
                    else:
                        for i in range(0,4):
                            for j in range(0,4):
                                s_x1.append(data[0:3,8:18,i*8:i*8+8,j*8:j*8+8]) #动态变量
                                s_x2.append(data[0:1,0:8,i*8:i*8+8,j*8:j*8+8]) #静态变量
    s_y = np.array(s_y) 
    s_x1 = np.array(s_x1)
    s_x2 = np.array(s_x2)
    s_x2 = np.squeeze(s_x2)
    s_y = np.nan_to_num(s_y, nan=0)
    s_x1 = np.nan_to_num(s_x1, nan=0)
    s_x2 = np.nan_to_num(s_x2, nan=0)
    # print(s_y.shape[0], s_y.shape, s_x1.shape, s_x2.shape)
    order = np.arange(s_y.shape[0])
    np.random.shuffle(order)
    train_y = []
    train_x1 = []
    train_x2 = []
    test_y = []
    test_x1 = []
    test_x2 = []
    vali_y = []
    vali_x1 = []
    vali_x2 = []
    for i in range(0,int(order.shape[0]*0.8)):
        # print(order[i])
        train_y.append(s_y[order[i]])
        train_x1.append(s_x1[order[i]])
        train_x2.append(s_x2[order[i]])
    for i in range(int(order.shape[0]*0.8),int(order.shape[0]*0.9)):
        # print(order[i])
        test_y.append(s_y[order[i]])
        test_x1.append(s_x1[order[i]])
        test_x2.append(s_x2[order[i]])
    for i in range(int(order.shape[0]*0.9),int(order.shape[0]*1.0)):
        # print(order[i])
        vali_y.append(s_y[order[i]])
        vali_x1.append(s_x1[order[i]])
        vali_x2.append(s_x2[order[i]])
    train_y = np.array(train_y)
    train_x1 = np.array(train_x1)
    train_x2 = np.array(train_x2)
    print('shape of original samples',train_x1.shape, train_x2.shape, train_y.shape)
    # train_x1 = train_x1.transpose(0,1,3,4,2)
    # train_x2 = train_x2.transpose(0,2,3,1)
    train_x2 = train_x2[:,None,:,:].copy()
    train_y = train_y[:,None,None,:,:].copy()
    test_y = np.array(test_y)
    test_x1 = np.array(test_x1)
    test_x2 = np.array(test_x2)
    # test_x1 = test_x1.transpose(0,1,3,4,2)
    # test_x2 = test_x2.transpose(0,2,3,1)
    test_x2 = test_x2[:,None,:,:].copy()
    test_y = test_y[:,None,None,:,:].copy()
    
    vali_y = np.array(vali_y)
    vali_x1 = np.array(vali_x1)
    vali_x2 = np.array(vali_x2)
    # vali_x1 = vali_x1.transpose(0,1,3,4,2)
    # vali_x2 = vali_x2.transpose(0,2,3,1)
    vali_x2 = vali_x2[:,None,:,:].copy()
    vali_y = vali_y[:,None,None,:,:].copy()
    
    print('shape of processed samples',train_y.shape,train_x1.shape,train_x2.shape,test_y.shape,test_x1.shape,test_x2.shape,vali_y.shape,vali_x1.shape,vali_x2.shape)
    return train_x1,train_x2,train_y,test_x1,test_x2,test_y,vali_x1,vali_x2,vali_y
def read_samples1603(rootdir):   
    s_y = []
    s_x1 = []  #动态变量
    s_x2 = []  #静态变量
    num=0
    for dirpath, filename, filenames in os.walk(rootdir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.h5':
                num+= 1
                if num%100 == 0:
                    print(num)
                # print (filename)                
                s_name = os.path.join(dirpath, filename)
                f = h5py.File(s_name,"r")
                for group in f.keys():
                    # print (group)
                    group_read = f[group]
                    data = np.array(group_read)
                    #print(data.shape)
                    if group =='y':
                        s_y.append(data[0:16,0:16])
                        s_y.append(data[0:16,16:32])
                        s_y.append(data[16:32,0:16])
                        s_y.append(data[16:32,16:32])
                    else:
                        s_x1.append(data[0:3,8:18,0:16,0:16]) #动态变量
                        s_x1.append(data[0:3,8:18,0:16,16:32]) #动态变量
                        s_x1.append(data[0:3,8:18,16:32,0:16]) #动态变量
                        s_x1.append(data[0:3,8:18,16:32,16:32]) #动态变量
                        s_x2.append(data[0:1,0:8,0:16,0:16]) #静态变量
                        s_x2.append(data[0:1,0:8,0:16,16:32]) #静态变量
                        s_x2.append(data[0:1,0:8,16:32,0:16]) #静态变量
                        s_x2.append(data[0:1,0:8,16:32,16:32]) #静态变量
    s_y = np.array(s_y) 
    s_x1 = np.array(s_x1)
    s_x2 = np.array(s_x2)
    s_x2 = np.squeeze(s_x2)
    s_y = np.nan_to_num(s_y, nan=0)
    s_x1 = np.nan_to_num(s_x1, nan=0)
    s_x2 = np.nan_to_num(s_x2, nan=0)
    # print(s_y.shape[0], s_y.shape, s_x1.shape, s_x2.shape)
    order = np.arange(s_y.shape[0])
    np.random.shuffle(order)
    train_y = []
    train_x1 = []
    train_x2 = []
    test_y = []
    test_x1 = []
    test_x2 = []
    vali_y = []
    vali_x1 = []
    vali_x2 = []
    for i in range(0,int(order.shape[0]*0.8)):
        # print(order[i])
        train_y.append(s_y[order[i]])
        train_x1.append(s_x1[order[i]])
        train_x2.append(s_x2[order[i]])
    for i in range(int(order.shape[0]*0.8),int(order.shape[0]*0.9)):
        # print(order[i])
        test_y.append(s_y[order[i]])
        test_x1.append(s_x1[order[i]])
        test_x2.append(s_x2[order[i]])
    for i in range(int(order.shape[0]*0.9),int(order.shape[0]*1.0)):
        # print(order[i])
        vali_y.append(s_y[order[i]])
        vali_x1.append(s_x1[order[i]])
        vali_x2.append(s_x2[order[i]])
    train_y = np.array(train_y)
    train_x1 = np.array(train_x1)
    train_x2 = np.array(train_x2)
    print('shape of original samples',train_x1.shape, train_x2.shape, train_y.shape)
    # train_x1 = train_x1.transpose(0,1,3,4,2)
    # train_x2 = train_x2.transpose(0,2,3,1)
    train_x2 = train_x2[:,None,:,:].copy()
    train_y = train_y[:,None,None,:,:].copy()
    test_y = np.array(test_y)
    test_x1 = np.array(test_x1)
    test_x2 = np.array(test_x2)
    # test_x1 = test_x1.transpose(0,1,3,4,2)
    # test_x2 = test_x2.transpose(0,2,3,1)
    test_x2 = test_x2[:,None,:,:].copy()
    test_y = test_y[:,None,None,:,:].copy()
    
    vali_y = np.array(vali_y)
    vali_x1 = np.array(vali_x1)
    vali_x2 = np.array(vali_x2)
    # vali_x1 = vali_x1.transpose(0,1,3,4,2)
    # vali_x2 = vali_x2.transpose(0,2,3,1)
    vali_x2 = vali_x2[:,None,:,:].copy()
    vali_y = vali_y[:,None,None,:,:].copy()
    
    print('shape of processed samples',train_y.shape,train_x1.shape,train_x2.shape,test_y.shape,test_x1.shape,test_x2.shape,vali_y.shape,vali_x1.shape,vali_x2.shape)
    return train_x1,train_x2,train_y,test_x1,test_x2,test_y,vali_x1,vali_x2,vali_y
def read_samples3203(rootdir):   
    s_y = []
    s_x1 = []  #动态变量
    s_x2 = []  #静态变量
    num=0
    for dirpath, filename, filenames in os.walk(rootdir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.h5':
                num+= 1
                if num%100 == 0:
                    print(num)
                # print (filename)                
                s_name = os.path.join(dirpath, filename)
                f = h5py.File(s_name,"r")
                for group in f.keys():
                    # print (group)
                    group_read = f[group]
                    data = np.array(group_read)
                    # print(data.shape)
                    if group =='y':
                        s_y.append(data)
                    else:
                        s_x1.append(data[0:3,8:18]) #动态变量
                        s_x2.append(data[0:1,0:8]) #静态变量
    s_y = np.array(s_y) 
    s_x1 = np.array(s_x1)
    s_x2 = np.array(s_x2)
    s_x2 = np.squeeze(s_x2)
    s_y = np.nan_to_num(s_y, nan=0)
    s_x1 = np.nan_to_num(s_x1, nan=0)
    s_x2 = np.nan_to_num(s_x2, nan=0)
    # print(s_y.shape[0], s_y.shape, s_x1.shape, s_x2.shape)
    order = np.arange(s_y.shape[0])
    np.random.shuffle(order)
    train_y = []
    train_x1 = []
    train_x2 = []
    test_y = []
    test_x1 = []
    test_x2 = []
    vali_y = []
    vali_x1 = []
    vali_x2 = []
    for i in range(0,int(order.shape[0]*0.8)):
        # print(order[i])
        train_y.append(s_y[order[i]])
        train_x1.append(s_x1[order[i]])
        train_x2.append(s_x2[order[i]])
    for i in range(int(order.shape[0]*0.8),int(order.shape[0]*0.9)):
        # print(order[i])
        test_y.append(s_y[order[i]])
        test_x1.append(s_x1[order[i]])
        test_x2.append(s_x2[order[i]])
    for i in range(int(order.shape[0]*0.9),int(order.shape[0]*1.0)):
        # print(order[i])
        vali_y.append(s_y[order[i]])
        vali_x1.append(s_x1[order[i]])
        vali_x2.append(s_x2[order[i]])
    train_y = np.array(train_y)
    train_x1 = np.array(train_x1)
    train_x2 = np.array(train_x2)
    print('shape of original samples',train_x1.shape, train_x2.shape, train_y.shape)
    # train_x1 = train_x1.transpose(0,1,3,4,2)
    # train_x2 = train_x2.transpose(0,2,3,1)
    train_x2 = train_x2[:,None,:,:].copy()
    train_y = train_y[:,None,None,:,:].copy()
    test_y = np.array(test_y)
    test_x1 = np.array(test_x1)
    test_x2 = np.array(test_x2)
    # test_x1 = test_x1.transpose(0,1,3,4,2)
    # test_x2 = test_x2.transpose(0,2,3,1)
    test_x2 = test_x2[:,None,:,:].copy()
    test_y = test_y[:,None,None,:,:].copy()
    
    vali_y = np.array(vali_y)
    vali_x1 = np.array(vali_x1)
    vali_x2 = np.array(vali_x2)
    # vali_x1 = vali_x1.transpose(0,1,3,4,2)
    # vali_x2 = vali_x2.transpose(0,2,3,1)
    vali_x2 = vali_x2[:,None,:,:].copy()
    vali_y = vali_y[:,None,None,:,:].copy()
    
    print('shape of processed samples',train_y.shape,train_x1.shape,train_x2.shape,test_y.shape,test_x1.shape,test_x2.shape,vali_y.shape,vali_x1.shape,vali_x2.shape)
    return train_x1,train_x2,train_y,test_x1,test_x2,test_y,vali_x1,vali_x2,vali_y
def read_samples0804(rootdir):   
    s_y = []
    s_x1 = []  #动态变量
    s_x2 = []  #静态变量
    num=0
    for dirpath, filename, filenames in os.walk(rootdir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.h5':
                num+= 1
                if num%100 == 0:
                    print(num)
                # print (filename)                
                s_name = os.path.join(dirpath, filename)
                f = h5py.File(s_name,"r")
                for group in f.keys():
                    # print (group)
                    group_read = f[group]
                    data = np.array(group_read)
                    #print(data.shape)
                    if group =='y':
                        for i in range(0,4):
                            for j in range(0,4):
                                s_y.append(data[i*8:i*8+8,j*8:j*8+8])
                                
                    else:
                        for i in range(0,4):
                            for j in range(0,4):
                                s_x1.append(data[0:4,8:18,i*8:i*8+8,j*8:j*8+8]) #动态变量
                                s_x2.append(data[0:1,0:8,i*8:i*8+8,j*8:j*8+8]) #静态变量
    s_y = np.array(s_y) 
    s_x1 = np.array(s_x1)
    s_x2 = np.array(s_x2)
    s_x2 = np.squeeze(s_x2)
    s_y = np.nan_to_num(s_y, nan=0)
    s_x1 = np.nan_to_num(s_x1, nan=0)
    s_x2 = np.nan_to_num(s_x2, nan=0)
    # print(s_y.shape[0], s_y.shape, s_x1.shape, s_x2.shape)
    order = np.arange(s_y.shape[0])
    np.random.shuffle(order)
    train_y = []
    train_x1 = []
    train_x2 = []
    test_y = []
    test_x1 = []
    test_x2 = []
    vali_y = []
    vali_x1 = []
    vali_x2 = []
    for i in range(0,int(order.shape[0]*0.8)):
        # print(order[i])
        train_y.append(s_y[order[i]])
        train_x1.append(s_x1[order[i]])
        train_x2.append(s_x2[order[i]])
    for i in range(int(order.shape[0]*0.8),int(order.shape[0]*0.9)):
        # print(order[i])
        test_y.append(s_y[order[i]])
        test_x1.append(s_x1[order[i]])
        test_x2.append(s_x2[order[i]])
    for i in range(int(order.shape[0]*0.9),int(order.shape[0]*1.0)):
        # print(order[i])
        vali_y.append(s_y[order[i]])
        vali_x1.append(s_x1[order[i]])
        vali_x2.append(s_x2[order[i]])
    train_y = np.array(train_y)
    train_x1 = np.array(train_x1)
    train_x2 = np.array(train_x2)
    print('shape of original samples',train_x1.shape, train_x2.shape, train_y.shape)
    # train_x1 = train_x1.transpose(0,1,3,4,2)
    # train_x2 = train_x2.transpose(0,2,3,1)
    train_x2 = train_x2[:,None,:,:].copy()
    train_y = train_y[:,None,None,:,:].copy()
    test_y = np.array(test_y)
    test_x1 = np.array(test_x1)
    test_x2 = np.array(test_x2)
    # test_x1 = test_x1.transpose(0,1,3,4,2)
    # test_x2 = test_x2.transpose(0,2,3,1)
    test_x2 = test_x2[:,None,:,:].copy()
    test_y = test_y[:,None,None,:,:].copy()
    
    vali_y = np.array(vali_y)
    vali_x1 = np.array(vali_x1)
    vali_x2 = np.array(vali_x2)
    # vali_x1 = vali_x1.transpose(0,1,3,4,2)
    # vali_x2 = vali_x2.transpose(0,2,3,1)
    vali_x2 = vali_x2[:,None,:,:].copy()
    vali_y = vali_y[:,None,None,:,:].copy()
    
    print('shape of processed samples',train_y.shape,train_x1.shape,train_x2.shape,test_y.shape,test_x1.shape,test_x2.shape,vali_y.shape,vali_x1.shape,vali_x2.shape)
    return train_x1,train_x2,train_y,test_x1,test_x2,test_y,vali_x1,vali_x2,vali_y
def read_samples1604(rootdir):   
    s_y = []
    s_x1 = []  #动态变量
    s_x2 = []  #静态变量
    num=0
    for dirpath, filename, filenames in os.walk(rootdir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.h5':
                num+= 1
                if num%100 == 0:
                    print(num)
                # print (filename)                
                s_name = os.path.join(dirpath, filename)
                f = h5py.File(s_name,"r")
                for group in f.keys():
                    # print (group)
                    group_read = f[group]
                    data = np.array(group_read)
                    #print(data.shape)
                    if group =='y':
                        s_y.append(data[0:16,0:16])
                        s_y.append(data[0:16,16:32])
                        s_y.append(data[16:32,0:16])
                        s_y.append(data[16:32,16:32])
                    else:
                        s_x1.append(data[0:4,8:18,0:16,0:16]) #动态变量
                        s_x1.append(data[0:4,8:18,0:16,16:32]) #动态变量
                        s_x1.append(data[0:4,8:18,16:32,0:16]) #动态变量
                        s_x1.append(data[0:4,8:18,16:32,16:32]) #动态变量
                        s_x2.append(data[0:1,0:8,0:16,0:16]) #静态变量
                        s_x2.append(data[0:1,0:8,0:16,16:32]) #静态变量
                        s_x2.append(data[0:1,0:8,16:32,0:16]) #静态变量
                        s_x2.append(data[0:1,0:8,16:32,16:32]) #静态变量
    s_y = np.array(s_y) 
    s_x1 = np.array(s_x1)
    s_x2 = np.array(s_x2)
    s_x2 = np.squeeze(s_x2)
    s_y = np.nan_to_num(s_y, nan=0)
    s_x1 = np.nan_to_num(s_x1, nan=0)
    s_x2 = np.nan_to_num(s_x2, nan=0)
    # print(s_y.shape[0], s_y.shape, s_x1.shape, s_x2.shape)
    order = np.arange(s_y.shape[0])
    np.random.shuffle(order)
    train_y = []
    train_x1 = []
    train_x2 = []
    test_y = []
    test_x1 = []
    test_x2 = []
    vali_y = []
    vali_x1 = []
    vali_x2 = []
    for i in range(0,int(order.shape[0]*0.8)):
        # print(order[i])
        train_y.append(s_y[order[i]])
        train_x1.append(s_x1[order[i]])
        train_x2.append(s_x2[order[i]])
    for i in range(int(order.shape[0]*0.8),int(order.shape[0]*0.9)):
        # print(order[i])
        test_y.append(s_y[order[i]])
        test_x1.append(s_x1[order[i]])
        test_x2.append(s_x2[order[i]])
    for i in range(int(order.shape[0]*0.9),int(order.shape[0]*1.0)):
        # print(order[i])
        vali_y.append(s_y[order[i]])
        vali_x1.append(s_x1[order[i]])
        vali_x2.append(s_x2[order[i]])
    train_y = np.array(train_y)
    train_x1 = np.array(train_x1)
    train_x2 = np.array(train_x2)
    print('shape of original samples',train_x1.shape, train_x2.shape, train_y.shape)
    # train_x1 = train_x1.transpose(0,1,3,4,2)
    # train_x2 = train_x2.transpose(0,2,3,1)
    train_x2 = train_x2[:,None,:,:].copy()
    train_y = train_y[:,None,None,:,:].copy()
    test_y = np.array(test_y)
    test_x1 = np.array(test_x1)
    test_x2 = np.array(test_x2)
    # test_x1 = test_x1.transpose(0,1,3,4,2)
    # test_x2 = test_x2.transpose(0,2,3,1)
    test_x2 = test_x2[:,None,:,:].copy()
    test_y = test_y[:,None,None,:,:].copy()
    
    vali_y = np.array(vali_y)
    vali_x1 = np.array(vali_x1)
    vali_x2 = np.array(vali_x2)
    # vali_x1 = vali_x1.transpose(0,1,3,4,2)
    # vali_x2 = vali_x2.transpose(0,2,3,1)
    vali_x2 = vali_x2[:,None,:,:].copy()
    vali_y = vali_y[:,None,None,:,:].copy()
    
    print('shape of processed samples',train_y.shape,train_x1.shape,train_x2.shape,test_y.shape,test_x1.shape,test_x2.shape,vali_y.shape,vali_x1.shape,vali_x2.shape)
    return train_x1,train_x2,train_y,test_x1,test_x2,test_y,vali_x1,vali_x2,vali_y
def read_samples3204(rootdir):   
    s_y = []
    s_x1 = []  #动态变量
    s_x2 = []  #静态变量
    num=0
    for dirpath, filename, filenames in os.walk(rootdir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.h5':
                num+= 1
                if num%100 == 0:
                    print(num)
                # print (filename)                
                s_name = os.path.join(dirpath, filename)
                f = h5py.File(s_name,"r")
                for group in f.keys():
                    # print (group)
                    group_read = f[group]
                    data = np.array(group_read)
                    # print(data.shape)
                    if group =='y':
                        s_y.append(data)
                    else:
                        s_x1.append(data[0:4,8:18]) #动态变量
                        s_x2.append(data[0:1,0:8]) #静态变量
    s_y = np.array(s_y) 
    s_x1 = np.array(s_x1)
    s_x2 = np.array(s_x2)
    s_x2 = np.squeeze(s_x2)
    s_y = np.nan_to_num(s_y, nan=0)
    s_x1 = np.nan_to_num(s_x1, nan=0)
    s_x2 = np.nan_to_num(s_x2, nan=0)
    # print(s_y.shape[0], s_y.shape, s_x1.shape, s_x2.shape)
    order = np.arange(s_y.shape[0])
    np.random.shuffle(order)
    train_y = []
    train_x1 = []
    train_x2 = []
    test_y = []
    test_x1 = []
    test_x2 = []
    vali_y = []
    vali_x1 = []
    vali_x2 = []
    for i in range(0,int(order.shape[0]*0.8)):
        # print(order[i])
        train_y.append(s_y[order[i]])
        train_x1.append(s_x1[order[i]])
        train_x2.append(s_x2[order[i]])
    for i in range(int(order.shape[0]*0.8),int(order.shape[0]*0.9)):
        # print(order[i])
        test_y.append(s_y[order[i]])
        test_x1.append(s_x1[order[i]])
        test_x2.append(s_x2[order[i]])
    for i in range(int(order.shape[0]*0.9),int(order.shape[0]*1.0)):
        # print(order[i])
        vali_y.append(s_y[order[i]])
        vali_x1.append(s_x1[order[i]])
        vali_x2.append(s_x2[order[i]])
    train_y = np.array(train_y)
    train_x1 = np.array(train_x1)
    train_x2 = np.array(train_x2)
    print('shape of original samples',train_x1.shape, train_x2.shape, train_y.shape)
    # train_x1 = train_x1.transpose(0,1,3,4,2)
    # train_x2 = train_x2.transpose(0,2,3,1)
    train_x2 = train_x2[:,None,:,:].copy()
    train_y = train_y[:,None,None,:,:].copy()
    test_y = np.array(test_y)
    test_x1 = np.array(test_x1)
    test_x2 = np.array(test_x2)
    # test_x1 = test_x1.transpose(0,1,3,4,2)
    # test_x2 = test_x2.transpose(0,2,3,1)
    test_x2 = test_x2[:,None,:,:].copy()
    test_y = test_y[:,None,None,:,:].copy()
    
    vali_y = np.array(vali_y)
    vali_x1 = np.array(vali_x1)
    vali_x2 = np.array(vali_x2)
    # vali_x1 = vali_x1.transpose(0,1,3,4,2)
    # vali_x2 = vali_x2.transpose(0,2,3,1)
    vali_x2 = vali_x2[:,None,:,:].copy()
    vali_y = vali_y[:,None,None,:,:].copy()
    
    print('shape of processed samples',train_y.shape,train_x1.shape,train_x2.shape,test_y.shape,test_x1.shape,test_x2.shape,vali_y.shape,vali_x1.shape,vali_x2.shape)
    return train_x1,train_x2,train_y,test_x1,test_x2,test_y,vali_x1,vali_x2,vali_y
def read_samples0805(rootdir):   
    s_y = []
    s_x1 = []  #动态变量
    s_x2 = []  #静态变量
    num=0
    for dirpath, filename, filenames in os.walk(rootdir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.h5':
                num+= 1
                if num%100 == 0:
                    print(num)
                # print (filename)                
                s_name = os.path.join(dirpath, filename)
                f = h5py.File(s_name,"r")
                for group in f.keys():
                    # print (group)
                    group_read = f[group]
                    data = np.array(group_read)
                    #print(data.shape)
                    if group =='y':
                        for i in range(0,4):
                            for j in range(0,4):
                                s_y.append(data[i*8:i*8+8,j*8:j*8+8])
                                
                    else:
                        for i in range(0,4):
                            for j in range(0,4):
                                s_x1.append(data[:,8:18,i*8:i*8+8,j*8:j*8+8]) #动态变量
                                s_x2.append(data[0:1,0:8,i*8:i*8+8,j*8:j*8+8]) #静态变量
    s_y = np.array(s_y) 
    s_x1 = np.array(s_x1)
    s_x2 = np.array(s_x2)
    s_x2 = np.squeeze(s_x2)
    s_y = np.nan_to_num(s_y, nan=0)
    s_x1 = np.nan_to_num(s_x1, nan=0)
    s_x2 = np.nan_to_num(s_x2, nan=0)
    # print(s_y.shape[0], s_y.shape, s_x1.shape, s_x2.shape)
    order = np.arange(s_y.shape[0])
    np.random.shuffle(order)
    train_y = []
    train_x1 = []
    train_x2 = []
    test_y = []
    test_x1 = []
    test_x2 = []
    vali_y = []
    vali_x1 = []
    vali_x2 = []
    for i in range(0,int(order.shape[0]*0.8)):
        # print(order[i])
        train_y.append(s_y[order[i]])
        train_x1.append(s_x1[order[i]])
        train_x2.append(s_x2[order[i]])
    for i in range(int(order.shape[0]*0.8),int(order.shape[0]*0.9)):
        # print(order[i])
        test_y.append(s_y[order[i]])
        test_x1.append(s_x1[order[i]])
        test_x2.append(s_x2[order[i]])
    for i in range(int(order.shape[0]*0.9),int(order.shape[0]*1.0)):
        # print(order[i])
        vali_y.append(s_y[order[i]])
        vali_x1.append(s_x1[order[i]])
        vali_x2.append(s_x2[order[i]])
    train_y = np.array(train_y)
    train_x1 = np.array(train_x1)
    train_x2 = np.array(train_x2)
    print('shape of original samples',train_x1.shape, train_x2.shape, train_y.shape)
    # train_x1 = train_x1.transpose(0,1,3,4,2)
    # train_x2 = train_x2.transpose(0,2,3,1)
    train_x2 = train_x2[:,None,:,:].copy()
    train_y = train_y[:,None,None,:,:].copy()
    test_y = np.array(test_y)
    test_x1 = np.array(test_x1)
    test_x2 = np.array(test_x2)
    # test_x1 = test_x1.transpose(0,1,3,4,2)
    # test_x2 = test_x2.transpose(0,2,3,1)
    test_x2 = test_x2[:,None,:,:].copy()
    test_y = test_y[:,None,None,:,:].copy()
    
    vali_y = np.array(vali_y)
    vali_x1 = np.array(vali_x1)
    vali_x2 = np.array(vali_x2)
    # vali_x1 = vali_x1.transpose(0,1,3,4,2)
    # vali_x2 = vali_x2.transpose(0,2,3,1)
    vali_x2 = vali_x2[:,None,:,:].copy()
    vali_y = vali_y[:,None,None,:,:].copy()
    
    print('shape of processed samples',train_y.shape,train_x1.shape,train_x2.shape,test_y.shape,test_x1.shape,test_x2.shape,vali_y.shape,vali_x1.shape,vali_x2.shape)
    return train_x1,train_x2,train_y,test_x1,test_x2,test_y,vali_x1,vali_x2,vali_y
def read_samples1605(rootdir):   
    s_y = []
    s_x1 = []  #动态变量
    s_x2 = []  #静态变量
    num=0
    for dirpath, filename, filenames in os.walk(rootdir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.h5':
                num+= 1
                if num%100 == 0:
                    print(num)
                # print (filename)                
                s_name = os.path.join(dirpath, filename)
                f = h5py.File(s_name,"r")
                for group in f.keys():
                    # print (group)
                    group_read = f[group]
                    data = np.array(group_read)
                    #print(data.shape)
                    if group =='y':
                        s_y.append(data[0:16,0:16])
                        s_y.append(data[0:16,16:32])
                        s_y.append(data[16:32,0:16])
                        s_y.append(data[16:32,16:32])
                    else:
                        s_x1.append(data[:,8:18,0:16,0:16]) #动态变量
                        s_x1.append(data[:,8:18,0:16,16:32]) #动态变量
                        s_x1.append(data[:,8:18,16:32,0:16]) #动态变量
                        s_x1.append(data[:,8:18,16:32,16:32]) #动态变量
                        s_x2.append(data[0:1,0:8,0:16,0:16]) #静态变量
                        s_x2.append(data[0:1,0:8,0:16,16:32]) #静态变量
                        s_x2.append(data[0:1,0:8,16:32,0:16]) #静态变量
                        s_x2.append(data[0:1,0:8,16:32,16:32]) #静态变量
    s_y = np.array(s_y) 
    s_x1 = np.array(s_x1)
    s_x2 = np.array(s_x2)
    s_x2 = np.squeeze(s_x2)
    s_y = np.nan_to_num(s_y, nan=0)
    s_x1 = np.nan_to_num(s_x1, nan=0)
    s_x2 = np.nan_to_num(s_x2, nan=0)
    # print(s_y.shape[0], s_y.shape, s_x1.shape, s_x2.shape)
    order = np.arange(s_y.shape[0])
    np.random.shuffle(order)
    train_y = []
    train_x1 = []
    train_x2 = []
    test_y = []
    test_x1 = []
    test_x2 = []
    vali_y = []
    vali_x1 = []
    vali_x2 = []
    for i in range(0,int(order.shape[0]*0.8)):
        # print(order[i])
        train_y.append(s_y[order[i]])
        train_x1.append(s_x1[order[i]])
        train_x2.append(s_x2[order[i]])
    for i in range(int(order.shape[0]*0.8),int(order.shape[0]*0.9)):
        # print(order[i])
        test_y.append(s_y[order[i]])
        test_x1.append(s_x1[order[i]])
        test_x2.append(s_x2[order[i]])
    for i in range(int(order.shape[0]*0.9),int(order.shape[0]*1.0)):
        # print(order[i])
        vali_y.append(s_y[order[i]])
        vali_x1.append(s_x1[order[i]])
        vali_x2.append(s_x2[order[i]])
    train_y = np.array(train_y)
    train_x1 = np.array(train_x1)
    train_x2 = np.array(train_x2)
    print('shape of original samples',train_x1.shape, train_x2.shape, train_y.shape)
    # train_x1 = train_x1.transpose(0,1,3,4,2)
    # train_x2 = train_x2.transpose(0,2,3,1)
    train_x2 = train_x2[:,None,:,:].copy()
    train_y = train_y[:,None,None,:,:].copy()
    test_y = np.array(test_y)
    test_x1 = np.array(test_x1)
    test_x2 = np.array(test_x2)
    # test_x1 = test_x1.transpose(0,1,3,4,2)
    # test_x2 = test_x2.transpose(0,2,3,1)
    test_x2 = test_x2[:,None,:,:].copy()
    test_y = test_y[:,None,None,:,:].copy()
    
    vali_y = np.array(vali_y)
    vali_x1 = np.array(vali_x1)
    vali_x2 = np.array(vali_x2)
    # vali_x1 = vali_x1.transpose(0,1,3,4,2)
    # vali_x2 = vali_x2.transpose(0,2,3,1)
    vali_x2 = vali_x2[:,None,:,:].copy()
    vali_y = vali_y[:,None,None,:,:].copy()
    
    print('shape of processed samples',train_y.shape,train_x1.shape,train_x2.shape,test_y.shape,test_x1.shape,test_x2.shape,vali_y.shape,vali_x1.shape,vali_x2.shape)
    return train_x1,train_x2,train_y,test_x1,test_x2,test_y,vali_x1,vali_x2,vali_y
def read_samples3205(rootdir):   
    s_y = []
    s_x1 = []  #动态变量
    s_x2 = []  #静态变量
    num=0
    for dirpath, filename, filenames in os.walk(rootdir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.h5':
                num+= 1
                if num%100 == 0:
                    print(num)
                # print (filename)                
                s_name = os.path.join(dirpath, filename)
                f = h5py.File(s_name,"r")
                for group in f.keys():
                    # print (group)
                    group_read = f[group]
                    data = np.array(group_read)
                    # print(data.shape)
                    if group =='y':
                        s_y.append(data)
                    else:
                        s_x1.append(data[:,8:18]) #动态变量
                        s_x2.append(data[0:1,0:8]) #静态变量
    s_y = np.array(s_y) 
    s_x1 = np.array(s_x1)
    s_x2 = np.array(s_x2)
    s_x2 = np.squeeze(s_x2)
    s_y = np.nan_to_num(s_y, nan=0)
    s_x1 = np.nan_to_num(s_x1, nan=0)
    s_x2 = np.nan_to_num(s_x2, nan=0)
    # print(s_y.shape[0], s_y.shape, s_x1.shape, s_x2.shape)
    order = np.arange(s_y.shape[0])
    np.random.shuffle(order)
    train_y = []
    train_x1 = []
    train_x2 = []
    test_y = []
    test_x1 = []
    test_x2 = []
    vali_y = []
    vali_x1 = []
    vali_x2 = []
    for i in range(0,int(order.shape[0]*0.8)):
        # print(order[i])
        train_y.append(s_y[order[i]])
        train_x1.append(s_x1[order[i]])
        train_x2.append(s_x2[order[i]])
    for i in range(int(order.shape[0]*0.8),int(order.shape[0]*0.9)):
        # print(order[i])
        test_y.append(s_y[order[i]])
        test_x1.append(s_x1[order[i]])
        test_x2.append(s_x2[order[i]])
    for i in range(int(order.shape[0]*0.9),int(order.shape[0]*1.0)):
        # print(order[i])
        vali_y.append(s_y[order[i]])
        vali_x1.append(s_x1[order[i]])
        vali_x2.append(s_x2[order[i]])
    train_y = np.array(train_y)
    train_x1 = np.array(train_x1)
    train_x2 = np.array(train_x2)
    print('shape of original samples',train_x1.shape, train_x2.shape, train_y.shape)
    # train_x1 = train_x1.transpose(0,1,3,4,2)
    # train_x2 = train_x2.transpose(0,2,3,1)
    train_x2 = train_x2[:,None,:,:].copy()
    train_y = train_y[:,None,None,:,:].copy()
    test_y = np.array(test_y)
    test_x1 = np.array(test_x1)
    test_x2 = np.array(test_x2)
    # test_x1 = test_x1.transpose(0,1,3,4,2)
    # test_x2 = test_x2.transpose(0,2,3,1)
    test_x2 = test_x2[:,None,:,:].copy()
    test_y = test_y[:,None,None,:,:].copy()
    
    vali_y = np.array(vali_y)
    vali_x1 = np.array(vali_x1)
    vali_x2 = np.array(vali_x2)
    # vali_x1 = vali_x1.transpose(0,1,3,4,2)
    # vali_x2 = vali_x2.transpose(0,2,3,1)
    vali_x2 = vali_x2[:,None,:,:].copy()
    vali_y = vali_y[:,None,None,:,:].copy()
    
    print('shape of processed samples',train_y.shape,train_x1.shape,train_x2.shape,test_y.shape,test_x1.shape,test_x2.shape,vali_y.shape,vali_x1.shape,vali_x2.shape)
    return train_x1,train_x2,train_y,test_x1,test_x2,test_y,vali_x1,vali_x2,vali_y

def test():
    outsum = []
    testmodel = TimesFormer_2input(configs).to(configs.device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    testmodel.to(device)
    model_info = torch.load(r'd:\DL\para-test\08-5-20241029\checkpoint_best.pth.tar')#改这里,加载最后一个checkpoint文件
    testmodel.load_state_dict(model_info['state_dict'])
    testset=ImageDataset_array(testX1,testX2,testY)
    testLoader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False)
    t = tqdm(testLoader, leave=False, total=len(testLoader))
    for i, imgs in enumerate(t):
        inputs1 = Variable(imgs["Ix1"]).to(device)  # B,S,C,H,W
        inputs2 = Variable(imgs["Ix2"]).to(device)
        inputs2 = torch.squeeze(inputs2,1)
        # inputs = Variable(imgs["Ix"]).to(device)  # B,S,C,H,W
        # label = Variable(imgs["Iy"]).to(device)  # B,S,C,H,W
        testoutput = testmodel(inputs1=inputs1,inputs2=inputs2,tgt=None, train=False) # B,S,C,H,W  
        # print(testoutput.shape)
        testoutput2 = testoutput.view(testoutput.shape[0], testoutput.shape[1], testoutput.shape[2],testoutput.shape[3]).cpu()
        result1 = testoutput2.detach().numpy()
        # print(result1.shape)
        outsum.append(result1)
    outsum = np.array(outsum)
    outsum = outsum.squeeze(axis=1).copy()
    print(outsum.shape)
    return outsum
def rmse(listpr,listin):
    n = 0
    for i in range(len(listpr)):
        er = pow(listpr[i]-listin[i],2)
        n += er
    rmse1 = sqrt(n/len(listpr))
    return rmse1
def Rnosquare(listpr,listin):
    SSpr = 0
    SSin = 0
    SS = 0
    meanpr = sum(listpr)/len(listpr)
    meanin = sum(listin)/len(listin)
    for item in listpr:
        erp = pow(meanpr-item,2)
        SSpr += erp
    for item in listin:
        eri = pow(meanin-item,2)
        SSin += eri
    for i,j in zip(listpr,listin):
        er = (i - meanpr)*(j - meanin)
        SS += er
        
    #Rsq = pow(SS,2)/(SSpr*SSin)     Rsquare
    Rsq = SS/((SSpr*SSin)** 0.5 )   
    return Rsq
if __name__ == "__main__":   
    print('start from here')
    # root_dir = 'f://samples_20231219' #sample_5
    root_dir = 'd://DL//samples_20241011' #sample_5
    (trainX1, trainX2, trainY,testX1, testX2, testY,valX1,valX2, valY)= read_samples0805(root_dir)    
    save_dir = r'd:\DL\para-test\08-5-20241029'  #改这里
    run_dir = r'd:\DL\para-test\08-5-20241029'  #改这里
    
    # trainset=ImageDataset_array(trainX1,trainX2,trainY)
    # validset=ImageDataset_array(valX1,valX2,valY)
    testset=ImageDataset_array(testX1,testX2,testY)
    print(testX1.shape, testX2.shape, testY.shape)
    
    # trainLoader = torch.utils.data.DataLoader(
    #         trainset,
    #         batch_size=configs.batch_size,
    #         shuffle=True)
    # validLoader = torch.utils.data.DataLoader(
    #         validset,
    #         batch_size=configs.batch_size_test,
    #         shuffle=True)
    testLoader = torch.utils.data.DataLoader(
            testset,
            batch_size=configs.batch_size_test,
            shuffle=False)
    
    # train()
    predY=test()
    
    testY = testY.squeeze().copy()
    testY = testY[:,:,:,None]
    print(testY.shape)
    RMSE1=[]
    CC=[]
    for i in range(0,len(testY)):
        testY_vali=np.ravel(testY[i])
        predY_vali=np.ravel(predY[i])
        RMSE1.append(rmse(predY_vali,testY_vali))
        CC.append(Rnosquare(predY_vali,testY_vali))
    RMSE1=np.array(RMSE1)
    RMSE1 = RMSE1.reshape(-1,1)
    RMSE1 = RMSE1.tolist()
    CC=np.array(CC)
    CC = CC.reshape(-1,1)
    CC = CC.tolist()
    print(RMSE1)
    csvfile_w = open(r'd:\DL\para-test\08-5-20241029\test-08-5-RMSE.csv', 'w',newline='')
    writer = csv.writer(csvfile_w)
    for i in range(0,len(RMSE1)):
        writer.writerow(RMSE1[i])
    csvfile_w.close()
    csvfile_w = open(r'd:\DL\para-test\08-5-20241029\test-08-5-R.csv', 'w',newline='')
    writer = csv.writer(csvfile_w)
    for i in range(0,len(CC)):
        writer.writerow(CC[i])
    csvfile_w.close()
    print('end at here')