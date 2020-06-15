#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, glob
import torch, torchvision
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset
from PIL import Image 


# In[2]:


def XMNIST_Dataloader(xmnist, root, tr, resize_32, num_samples, batch_size):
    use_cuda = True
    num_workers = 7
    kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
    if resize_32:
        trans = torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                # torchvision.transforms.Pad(2),
                torchvision.transforms.ToTensor()])
    else: 
        trans = torchvision.transforms.ToTensor()
    
    if xmnist == 'MNIST':
        data = torchvision.datasets.MNIST(root=root, train=tr, transform=trans, download=False)
    elif xmnist == 'KMNIST':
        data = torchvision.datasets.KMNIST(root=root, train=tr, transform=trans, download=False)
    elif xmnist == 'QMNIST':
        data = torchvision.datasets.QMNIST(root=root, train=tr, transform=trans, download=False)
    elif xmnist == 'FashionMNIST':
        data = torchvision.datasets.FashionMNIST(root=root, train=tr, transform=trans, download=False)    
    else:
        raise ValueError("Only support: MNIST, KMNIST, QMNIST, FashionMNIST")
        
    if num_samples > 0:
        loader = Data.DataLoader(dataset=data, 
                                 batch_size=batch_size, 
                                 shuffle=False,
                                 sampler=Data.sampler.RandomSampler(data, num_samples=num_samples, replacement=True))
    else:
        loader = Data.DataLoader(dataset=data, 
                                 batch_size=batch_size, 
                                 shuffle=True)
    return loader


# In[3]:


def EMNIST_dataloader(xmnist, root, resize_32, num_samples, batch_size):
    use_cuda = True
    num_workers = 7
    kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
    
    if resize_32:
        trans = torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor()])
    else: 
        trans = torchvision.transforms.ToTensor()
        
    data = torchvision.datasets.EMNIST(root=root,
                                       split=xmnist,
                                       transform=trans)
    
    loader = Data.DataLoader(dataset=data, 
                                 batch_size=batch_size, 
                                 shuffle=False,
                                 sampler=Data.sampler.RandomSampler(data, num_samples=num_samples, replacement=True))
    return loader


# In[4]:


def Omniglot_dataloader(root, resize_32, num_samples, batch_size):
    use_cuda = True
    num_workers = 7
    kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
    
    if resize_32:
        trans = torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor()])
    else: 
        trans = torchvision.transforms.Compose([
                torchvision.transforms.Resize((28, 28)),
                torchvision.transforms.ToTensor()])
        
    data = torchvision.datasets.Omniglot(root=root,
                                         transform=trans)
    
    loader = Data.DataLoader(dataset=data, 
                                 batch_size=batch_size, 
                                 shuffle=False,
                                 sampler=Data.sampler.RandomSampler(data, num_samples=num_samples, replacement=True))
    return loader


# In[5]:


def Noise_dataloader(distribution, resize_32, num_samples, batch_size):
    use_cuda = True
    num_workers = 7
    kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
    if resize_32:
        size = 32
    else:
        size = 28
        
    if distribution == 'uniform':
        data = torch.rand(size=(num_samples, 1, size, size))
    elif distribution == 'Gaussian':
        data = torch.randn(size=(num_samples, 1, size, size)) + 0.5
        data = clip_by_tensor(data, 0, 1)
    else:
        raise ValueError("Only support: uniform, Gaussian")

    loader = Data.DataLoader(data,
                             batch_size=batch_size, 
                             **kwargs)
    return loader

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    # t=t.float()
    # t_min=t_min.float()
    # t_max=t_max.float()
 
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


# In[6]:


class Half_XMNIST_Dataset(Dataset):
    def __init__(self, root, mode, mini, resize_32):
        self.root = root
        self.mode = mode
        self.mini = mini
        self.resize_32 = resize_32
        if self.resize_32:
            self.transform = torchvision.transforms.Compose([
                             torchvision.transforms.Resize((32, 32)),
                             torchvision.transforms.ToTensor()])
        else:
            self.transform = torchvision.transforms.ToTensor()

        if self.mini == 40:
            self.img_list = glob.glob(os.path.join(self.root, self.mode, '[0-4]/*.png'))
        elif self.mini == 59:
            self.img_list = glob.glob(os.path.join(self.root, self.mode, '[5-9]/*.png'))
        else:
            raise ValueError("Only support: 40, 59")
        
    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        target = int(self.img_list[index].split('/')[-2])
        return self.transform(img), target
 
    def __len__(self):
        return len(self.img_list)


# In[7]:


def Half_XMNIST_Dataloader(data_name, batch_size, mode, mini, resize_32):
    
    num_workers = 7
    use_cuda = True
    kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
    root = os.path.join('/4T/ood/nlood/torch/data', data_name, 'raw/')
    dataset = Half_XMNIST_Dataset(root, mode, mini, resize_32)
    
    loader = Data.DataLoader(dataset,
                             batch_size=batch_size, 
                             shuffle=True,
                             **kwargs)
    return loader


# In[8]:


def NotMNIST_dataloader(tr, resize_32, num_samples, batch_size):
    use_cuda = True
    num_workers = 7
    kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
    if resize_32:
        trans = torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor()])
    else: 
        trans = torchvision.transforms.ToTensor()
    

    data = torchvision.datasets.MNIST(root='/4T/ml_dataset/torch_data/NotMNIST', 
                                      train=tr, 
                                      transform=trans, 
                                      download=False)

    if num_samples > 0:
        loader = Data.DataLoader(dataset=data, 
                                 batch_size=batch_size, 
                                 shuffle=False,
                                 sampler=Data.sampler.RandomSampler(data, num_samples=num_samples, replacement=True))
    else:
        loader = Data.DataLoader(dataset=data, 
                                 batch_size=batch_size, 
                                 shuffle=True)
    return loader


