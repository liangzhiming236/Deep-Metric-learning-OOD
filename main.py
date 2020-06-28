#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import net
import matplotlib.pyplot as plt
import Data_loader

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score


# In[2]:


# settings
parser = argparse.ArgumentParser(description='Train code')
parser.add_argument('--batchsize', type=int, default=256, help='input batch size for test')
parser.add_argument('--epochs', type=int, default=100, help='number of epoch to train')
parser.add_argument('--resize', type=bool, default=False, help='resize to 32')
parser.add_argument('--ID', type=str, default='NotMNIST', help='the ID data')
parser.add_argument('--lr', type=float, default=0.001, help='the learning rate')
parser.add_argument('--margin', type=float, default=100, help='the margin between ID and OOD')
parser.add_argument('--beta', type=float, default=1, help='the')

args = parser.parse_args(args=[])
print (args)


# In[3]:


ae = net.autoencoder2()
ae.cuda()
optimizer_AE = torch.optim.Adam(ae.parameters(), lr=args.lr)


# In[4]:


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
        resize_32=args.resize,
        num_samples=-1,
        batch_size=args.batchsize,
    )

    test_loader = Data_loader.XMNIST_Dataloader(
        args.ID,
        root='/4T/ml_dataset/torch_data',
        tr=False,
        resize_32=args.resize,
        num_samples=-1,
        batch_size=args.batchsize,
    )


# In[5]:


Gaussian_noise_loader = Data_loader.Noise_dataloader(28, 'Gaussian', args.batchsize)

Uniform_noise_loader = Data_loader.Noise_dataloader(28, 'Uniform', args.batchsize)

FashionMNIST_loader = Data_loader.XMNIST_Dataloader(
    'FashionMNIST',
    root='/4T/ml_dataset/torch_data',
    tr=False,
    resize_32=False,
    num_samples=-1,
    batch_size=args.batchsize,
)

MNIST_loader = Data_loader.XMNIST_Dataloader(
    'MNIST',
    root='/4T/ml_dataset/torch_data',
    tr=False,
    resize_32=False,
    num_samples=-1,
    batch_size=args.batchsize,
)

KMNIST_loader = Data_loader.XMNIST_Dataloader(
    'KMNIST',
    root='/4T/ml_dataset/torch_data',
    tr=False,
    resize_32=False,
    num_samples=-1,
    batch_size=args.batchsize,
)

NotMNIST_loader = Data_loader.NotMNIST_dataloader(tr=False, 
                                      resize_32=False, 
                                      num_samples=-1, 
                                      batch_size=args.batchsize)

Omniglot_loader = Data_loader.Omniglot_dataloader(root='/4T/ml_dataset/torch_data', 
                                      resize_32=False, 
                                      num_samples=10000, 
                                      batch_size=args.batchsize)

EMNIST_letter_loader = Data_loader.EMNIST_dataloader(
    xmnist='letters', 
    root='/4T/ml_dataset/torch_data', 
    resize_32=False, 
    num_samples=10000,
    batch_size=args.batchsize)


# In[6]:


model = net.Net()
model.cuda()


# In[8]:


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


# In[9]:


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


# In[10]:


class ood_contrastive_loss(torch.nn.Module):
    def __init__(self, margin=100, beta=1):
        super(ood_contrastive_loss, self).__init__()
        self.margin = margin
        self.beta = beta
        
    def forward(self, embeddings, target):
        dist = euclidean_dist(embeddings, embeddings)
        ID_ID_dist = dist[0: len(dist)//2, 0: len(dist)//2]
        ID_OOD_dist = dist[len(dist)//2: , 0: len(dist)//2]
        
        ID_embeddings = embeddings[0: len(dist)//2]
        OOD_embeddings = embeddings[len(dist)//2: ]
        
        origin = torch.zeros(size=ID_embeddings.shape).cuda()
        ID_origin_dist = euclidean_dist(embeddings, origin)
        
        ID_target = target[0: len(dist)//2].float()
        ID_target_dist = euclidean_dist( ID_target.reshape(-1,1), ID_target.reshape(-1,1)).int()
        y = torch.where(ID_target_dist>0, torch.tensor(1).cuda(), torch.tensor(0).cuda())
        
        loss_ID = ((1-y)*ID_ID_dist**2 + y*torch.clamp(self.margin - ID_ID_dist, min=0.0)**2)/2
        loss_ID = torch.mean(loss_ID)
        
        loss_OOD = torch.clamp(self.beta*self.margin-ID_OOD_dist, min=0.0)
        loss_OOD = torch.mean(loss_OOD)
        
        loss_origin = torch.mean(ID_origin_dist)
        
        loss = loss_ID + loss_OOD + loss_origin

        return loss, loss_ID, loss_OOD, loss_origin


# In[11]:


def val():
    model.eval
    IDs = []
    oods = []
    for x, _ in train_loader:
        x = x.cuda()
        ood = model(ae(x))
        ID = model(x)
        IDs.append(ID.cpu().detach().numpy())
        oods.append(ood.cpu().detach().numpy())
    IDs = np.concatenate(IDs)
    oods = np.concatenate(oods)
    
    centers = cal_center(train_loader, model)
    
    preds = np.concatenate((IDs, oods))
    #     ys = np.concatenate(ys)
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
#     accuracy = 0
#     for threshold in thresholds:
#         y_pred = np.where(y_score>threshold, 1, 0)
#         if accuracy_score(y_true, y_pred) > accuracy:
#             accuracy = accuracy_score(y_true, y_pred)
#     error = (1-accuracy) * 100

#     # AUPR_in   
#     precision, recall, _ = precision_recall_curve(1-y_true, np.max(y_score)-y_score, pos_label=1)
#     AUPR_in = auc(recall, precision) * 100

#     # AUPR_out
#     precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=1)
#     AUPR_out = auc(recall, precision) * 100

    print ('FPR95', round(fpr95, 2), 
#            '\nError', round(error, 2),
           '\nAUROC', round(auroc, 2),
#            '\nAUPR_out', round(AUPR_out, 2),
#            '\nAUPR_in', round(AUPR_in, 2)
          )
    return fpr95, auroc, #error,  AUPR_out, AUPR_in


# In[12]:


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


# In[13]:


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_func = nn.CrossEntropyLoss()


# In[ ]:


loss_all_list = []
roc_list = []
lossOOD_list = []
loss_gen_list = []

cumulative_num = 0
init_value = 0
for epoch in range(args.epochs):
    model.train()
#     adjust_learning_rate(optimizer, epoch)
    print ('-'*20)
    print ('Epoch', epoch)
    for step, (x, y) in enumerate(train_loader):
        
        b_x = x.cuda()
        b_y = y.cuda()
        x_mini = b_x
        #####################################
        #          1. Update model
        #####################################

        ood_samples_mini = ae(x_mini.detach())
        
#         ood_samples_mini = ood_samples_mini + (torch.randn(size=ood_samples_mini.shape)/5).cuda()
        
        ood_target = torch.zeros(size=(len(ood_samples_mini), ), dtype=int) + 10
        ood_target = ood_target.cuda()

        data_and_ood = torch.cat((b_x, ood_samples_mini))
        target_and_ood = torch.cat((b_y, ood_target))
        shuffle = np.random.permutation(range(len(data_and_ood)))
        data_and_ood = data_and_ood[shuffle]
        target_and_ood = target_and_ood[shuffle]

        output = model(data_and_ood)
        
        target_sort, sort_index = torch.sort(target_and_ood)
        output_sort = output[sort_index]
        
        criteria_model = ood_contrastive_loss(margin=args.margin, beta=args.beta)
        loss, lossID, loss_OOD, loss_origin = criteria_model(output, target_and_ood)
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        #####################################
        #          2. Update autoencoder
        #####################################
        ood_samples = ae(b_x.detach())
 
        eleven = model(ood_samples)
        criterion_generator = nn.MSELoss()
        generator_loss = criterion_generator(b_x, ood_samples)
        
        ae_loss = generator_loss + 1*loss_OOD
        
        optimizer_AE.zero_grad()
        ae_loss.backward(retain_graph=True)
        optimizer_AE.step()
        

    
    loss_all_list.append(float(loss))
    
    lossOOD_list.append(float(loss_OOD))
    loss_gen_list.append(float(generator_loss))
     
    print ('loss', loss.detach())
    print ('lossID', lossID.detach())
    print ('lossOOD', loss_OOD.detach())
    print ('loss_gen', generator_loss.detach())
    print ('loss_origin', loss_origin.detach())    
        
#   test
#     center = cal_center(train_loader, model)
    result = val()[1]
    roc_list.append(result)
    torch.save(model, './checkpoint/ID_{}_checkpoint/model_{}'.format(args.ID, epoch))
    torch.save(ae, './checkpoint/ID_{}_checkpoint/autoencoder_{}'.format(args.ID, epoch))
#   earlystop
    
#     if result < init_value:
#         cumulative_num += 1
#     else:
#         init_value = result
#         cumulative_num = 0
#     if cumulative_num == 10:
#         print ('Early stop! Epoch', epoch)
#         break
    


# In[ ]:


np.save('./checkpoint/ID_{}_checkpoint/roc_list.npy'.format(args.ID), roc_list)


# In[ ]:


def test(id_loader_te, ood_loader_te, model, centers):
    model.eval()
    preds = []
    ys = []
    for x, y in id_loader_te:
        b_x = x.cuda()
        pred = model(b_x)
        pred = pred.cpu().detach().numpy()
        preds.append(pred)
        ys.append(y.numpy())

    for x, _ in ood_loader_te:
        b_x = x.cuda()
        pred = model(b_x)
        pred = pred.cpu().detach().numpy()
        preds.append(pred)

    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
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


# In[ ]:


loss_list = np.load('/4T/ood/zzzz/checkpoint/ID_{}_checkpoint/roc_list.npy'.format(args.ID))
n = np.argmax(loss_list)
# n = 1
print (n)
best_model = torch.load('./checkpoint/ID_{}_checkpoint/model_{}'.format(args.ID, n))
center = cal_center(train_loader, best_model)


# In[ ]:


OOD_list = [MNIST_loader, KMNIST_loader, FashionMNIST_loader, NotMNIST_loader,
            Omniglot_loader, EMNIST_letter_loader, 
            Gaussian_noise_loader, Uniform_noise_loader]


# In[ ]:


result_npy = np.zeros(shape=(8, 5))
for i, ood_loader in enumerate(OOD_list):
    print ('-'*10, ood_loader, '-'*10)
    result = test(test_loader, ood_loader, best_model, center)
    for j in range(5):
        result_npy[i, j] = result[j]
np.savetxt('./result/{}_result.csv'.format(args.ID), result_npy, delimiter = ',')



