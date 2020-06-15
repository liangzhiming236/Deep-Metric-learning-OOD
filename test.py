#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import vgg, net
import matplotlib.pyplot as plt
import Data_loader

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score


# In[2]:


# settings
parser = argparse.ArgumentParser(description='Test code')
parser.add_argument('--batchsize', type=int, default=256, help='input batch size for test')
parser.add_argument('--epoch', type=int, default=0, help='number of epoch to load')
parser.add_argument('--resize', type=bool, default=False, help='resize to 32')
parser.add_argument('--ID', type=str, default='NotMNIST', help='the ID data')

args = parser.parse_args()


print (args.ID)


if args.ID == 'NotMNIST':
    train_loader = Data_loader.NotMNIST_dataloader(
        tr=True,
        resize_32=args.resize,
        num_samples=-1,
        batch_size=args.batchsize,
    )

    test_loader = Data_loader.NotMNIST_dataloader(
        tr=False,
        resize_32=args.resize,
        num_samples=-1,
        batch_size=args.batchsize,
    )
else:
    train_loader = Data_loader.XMNIST_Dataloader(
        args.ID,
        root='/4T/ml_dataset/torch_data',
        tr=True,
        resize_32=False,
        num_samples=-1,
        batch_size=args.batchsize,
    )

    test_loader = Data_loader.XMNIST_Dataloader(
        args.ID,
        root='/4T/ml_dataset/torch_data',
        tr=False,
        resize_32=False,
        num_samples=-1,
        batch_size=args.batchsize,
    )


# In[5]:


Gaussian_noise_loader = Data_loader.Noise_dataloader('Gaussian', 
                                                     resize_32=False,
                                                     num_samples=10000, 
                                                     batch_size=128)


# In[6]:


Uniform_noise_loader = Data_loader.Noise_dataloader('uniform', 
                                                     resize_32=False,
                                                     num_samples=10000, 
                                                     batch_size=128)


# In[7]:


FashionMNIST_loader = Data_loader.XMNIST_Dataloader(
    'FashionMNIST',
    root='/4T/ml_dataset/torch_data',
    tr=False,
    resize_32=False,
    num_samples=-1,
    batch_size=args.batchsize,
)


# In[8]:


MNIST_loader = Data_loader.XMNIST_Dataloader(
    'MNIST',
    root='/4T/ml_dataset/torch_data',
    tr=False,
    resize_32=False,
    num_samples=-1,
    batch_size=args.batchsize,
)


# In[9]:


KMNIST_loader = Data_loader.XMNIST_Dataloader(
    'KMNIST',
    root='/4T/ml_dataset/torch_data',
    tr=False,
    resize_32=False,
    num_samples=-1,
    batch_size=args.batchsize,
)


# In[10]:


NotMNIST_loader = Data_loader.NotMNIST_dataloader(tr=False, 
                                      resize_32=False, 
                                      num_samples=-1, 
                                      batch_size=args.batchsize)


# In[11]:


Omniglot_loader = Data_loader.Omniglot_dataloader(root='/4T/ml_dataset/torch_data', 
                                      resize_32=False, 
                                      num_samples=10000, 
                                      batch_size=args.batchsize)


# In[12]:


EMNIST_letter_loader = Data_loader.EMNIST_dataloader(
    xmnist='letters', 
    root='/4T/ml_dataset/torch_data', 
    resize_32=False, 
    num_samples=10000,
    batch_size=args.batchsize)


# In[13]:


def cal_center(data_loader, model):
    embeddings = []
    ys = []
    for x, y in data_loader:
        b_x = x.cuda()
        pred = model(b_x)
        pred = pred.cpu().detach().numpy()
        embeddings.append(pred)
        ys.append(y.cpu().numpy())
    embeddings = np.concatenate(embeddings)
    ys = np.concatenate(ys)

    num_classes = len(set(ys))
    centers = np.zeros(shape=(num_classes, 64))
    for i in range(num_classes):
        index = np.where(ys==i)[0]
        center = np.mean(embeddings[index], axis=0)
        centers[i] = center
    return centers


# In[14]:


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


# In[15]:


def test(id_loader_te, ood_loader_te, model, centers, noise=False):
    model.eval()
    preds = []
    ys = []
    for x, y in id_loader_te:
        b_x = x.cuda()
        pred = model(b_x)
        pred = pred.cpu().detach().numpy()
        preds.append(pred)
        ys.append(y.numpy())

    for xy in ood_loader_te:
        if noise:
            x = xy
        else:
            x = xy[0]
        b_x = x.cuda()
        pred = model(b_x)
        pred = pred.cpu().detach().numpy()
        preds.append(pred)

    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
#     print (centers.shape)
    d = euclidean_distances(preds, centers)
    y_score = np.min(d, axis=1)
    y_true = np.zeros(shape=y_score.shape)
    y_true[len(y_true)//2: ] = 1
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # AUROC
    auroc = auc(fpr, tpr) * 100

    # FPR95
    fpr95 = fpr[np.argmin(np.abs(tpr - 0.95))] * 100

    # Detection error
    accuracy = 0
    for threshold in thresholds:
        y_pred = np.where(y_score>threshold, 1, 0)
        if accuracy_score(y_true, y_pred) > accuracy:
            accuracy = accuracy_score(y_true, y_pred)
    error = (1-accuracy) * 100

    # AUPR_in   
    precision, recall, _ = precision_recall_curve(1-y_true, np.max(y_score)-y_score, pos_label=1)
    AUPR_in = auc(recall, precision) * 100

    # AUPR_out
    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=1)
    AUPR_out = auc(recall, precision) * 100

    print ('FPR95', round(fpr95, 2), 
           '\nError', round(error, 2),
           '\nAUROC', round(auroc, 2),
           '\nAUPR_out', round(AUPR_out, 2),
           '\nAUPR_in', round(AUPR_in, 2))
    return fpr95, error, auroc, AUPR_out, AUPR_in


# In[16]:


def test_all(model, OOD_list):
    OOD_noise_list = OOD_list[-2:]
    OOD_list = OOD_list[0: -2]
    for ood_loader in OOD_list:
        print ('-'*10, ood_loader, '-'*10)
        test(test_loader, ood_loader, best_model, center, noise=False)
    for ood_loader in OOD_noise_list:
        print ('-'*10, ood_loader, '-'*10)
        test(test_loader, ood_loader, best_model, center, noise=True)


# In[17]:


OOD_list = [MNIST_loader, KMNIST_loader, FashionMNIST_loader, NotMNIST_loader,
            Omniglot_loader, EMNIST_letter_loader, 
            Gaussian_noise_loader, Uniform_noise_loader]


# In[ ]:


if args.epoch == 0:
    loss_list = np.load('/4T/ood/zzzz/checkpoint/ID_{}_checkpoint/lossID_list.npy'.format(args.ID))
    n = np.argmin(loss_list)
    best_model = torch.load('./checkpoint/ID_{}_checkpoint/model_{}'.format(args.ID, n))
    center = cal_center(train_loader, best_model)
else:
    best_model = torch.load('./checkpoint/ID_{}_checkpoint/model_{}'.format(args.ID, args.epoch))
    center = cal_center(train_loader, best_model)


# In[22]:


test_all(best_model, OOD_list)


# In[ ]:




