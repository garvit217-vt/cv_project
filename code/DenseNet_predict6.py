#!/usr/bin/env python
# coding: utf-8

# In[147]:


import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import os
from PIL import Image
from PIL.ImageOps import autocontrast
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np
from PIL import ImageFile
#from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import pandas as pd
import os
import random
from shutil import copyfile
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import re
import albumentations as albu
from albumentations.pytorch import ToTensor
# from catalyst.data import Augmentor
#import Augmentor
#import torchxrayvision as xrv


# In[2]:


import torch
import torchvision
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.optim as optimimg_path
from torch.utils.data import Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
import re
import albumentations as albu
from albumentations.pytorch import ToTensor
# from catalyst.data import Augmentor
#import Augmentor
# from imageio import imread, imsave
from skimage.io import imread, imsave
import skimage
import scipy.misc as m

torch.cuda.empty_cache()



# In[2]:


# get_ipython().system('pip install --upgrade efficientnet-pytorch')


# In[3]:

# normalize = transforms.Normalize(mean=0.5, std=0.5)
#normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#normalize = transforms.Normalize(mean=[0.5], std=[0.5])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transformer = transforms.Compose([
    transforms.Resize(256),
    #transforms.RandomResizedCrop((256),scale=(0.5,1.0)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize(256),
    # transforms.Resize(224),
    #transforms.CenterCrop(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


# In[80]:


batchsize = 1#10

def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=None):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        File structure:
        - root_dir
            - CT_COVID
                - img1.png
                - img2.png
                - ......
            - CT_NonCOVID
                - img1.png
                - img2.png
                - ......
        """
        self.root_dir = root_dir
        self.txt_path = [txt_COVID, txt_NonCOVID]
        self.classes = ['positive/', 'negative/']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir,self.classes[c],item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list

        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        img_path = self.img_list[idx][0]
        image = Image.open(img_path,'r')

        image = image.convert('F')

        image_np = np.array(image)

        rmax = np.amax(image_np)
        rmin = np.amin(image_np)
        
        image_np = 0 + ((image_np - rmin)/(rmax - rmin)*(255-0))

        image = Image.fromarray(image_np)
        image = image.convert('L')


        #print("10, 10: ", image.getpixel((10, 10)))
        #print("100, 100: ", image.getpixel((100, 100)))

        # image = skimage.img_as_float(image)
        # rgb_batch = np.repeat(image[..., np.newaxis],3,-1)
        # img = np.ascontiguousarray(rgb_batch.transpose(2,0,1))
        # image = Image.fromarray(img,mode='RGB')

        w, h = image.size
        im = Image.new('RGB', (w,h))
        data = zip(image.getdata(), image.getdata(), image.getdata())
        im.putdata(list(data))
        #print("10, 10: ", im.getpixel((10, 10)))
        #print("101, 101: ", im.getpixel((100, 101)))

        image = image.convert('F')
        
        if self.transform:
            #print("transform")
            #im = self.transform(image)
            im = self.transform(im)

        #print("max", torch.max(im))
        #print("min", torch.min(im))
        #print("mean", torch.mean(im))

        sample = {'img': im,
                  'label': int(self.img_list[idx][1])}
        return sample




if __name__ == '__main__':
    trainset = CovidCTDataset(root_dir='/home/garvit/analysis_ai_data1/',
                              txt_COVID='/home/garvit/analysis_ai_data1/np_train.txt',
                              txt_NonCOVID='/home/garvit/analysis_ai_data1/ncp_train.txt',
                              transform= train_transformer)
    valset = CovidCTDataset(root_dir='/home/garvit/analysis_ai_data1/',
                              txt_COVID='/home/garvit/analysis_ai_data1/np_val.txt',
                              txt_NonCOVID='/home/garvit/analysis_ai_data1/ncp_val.txt',
                              transform= val_transformer)
    testset = CovidCTDataset(root_dir='/home/garvit/analysis_ai_data1/',
                              txt_COVID='/home/garvit/analysis_ai_data1/np_test.txt',
                              txt_NonCOVID='/home/garvit/analysis_ai_data1/ncp_test.txt',
                              transform= val_transformer)
#
    #print(trainset.__len__())
    #print(valset.__len__())
    #print(testset.__len__())

    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True, num_workers= 0)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=True, num_workers= 0)
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=True, num_workers= 0)


# In[31]:


# for batch_index, batch_samples in enumerate(train_dataloader):
#         data, target = batch_samples[0], batch_samples[1]
# skimage.io.imshow(data[0,1,:,:].numpy())


# In[154]:

alpha = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(optimizer, epoch):

    model.train()

    train_loss = 0
    train_correct = 0

    for batch_index, batch_samples in enumerate(train_loader):

        # move data to device
        data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
        #data, target = batch_samples['img'], batch_samples['label']
#        data = data[:, 0, :, :]
#        data = data[:, None, :, :]
#         data, targets_a, targets_b, lam = mixup_data(data, target.long(), alpha, use_cuda=True)


        optimizer.zero_grad()
        output = model(data)

        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, target.long())
#         loss = mixup_criterion(criteria, output, targets_a, targets_b, lam)
        train_loss += criteria(output, target.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()

        # Display progress and write to tensorboard
        if batch_index % bs == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item()/ bs))
            
            f = open('model_result/{}_train_loss.txt'.format(modelname), 'a+')
            f.write(loss.item()/ bs)
            f.write(' ')
            f.close()



    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss/len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))
    f = open('model_result/{}.txt'.format(modelname), 'a+')
    f.write('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss/len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))
    f.write('\n')
    f.close()


# In[155]:


def val(epoch):

    model.eval()
    test_loss = 0
    correct = 0
    results = []

    TP = 0
    TN = 0
    FN = 0
    FP = 0


    criteria = nn.CrossEntropyLoss()
    # Don't update model
    with torch.no_grad():
        tpr_list = []
        fpr_list = []

        predlist=[]
        scorelist=[]
        targetlist=[]
        # Predict
        for batch_index, batch_samples in enumerate(val_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            #data, target = batch_samples['img'], batch_samples['label']
#            data = data[:, 0, :, :]
#            data = data[:, None, :, :]
            output = model(data)

            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
#             print('target',target.long()[:, 2].view_as(pred))
            correct += pred.eq(target.long().view_as(pred)).sum().item()

#             print(output[:,1].cpu().numpy())
#             print((output[:,1]+output[:,0]).cpu().numpy())
#             predcpu=(output[:,1].cpu().numpy())/((output[:,1]+output[:,0]).cpu().numpy())
            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)


    return targetlist, scorelist, predlist


# In[152]:


def test(epoch):

    model.eval()
    test_loss = 0
    correct = 0
    results = []

    TP = 0
    TN = 0
    FN = 0
    FP = 0


    criteria = nn.CrossEntropyLoss()
    # Don't update model
    with torch.no_grad():
        tpr_list = []
        fpr_list = []

        predlist=[]
        scorelist=[]
        targetlist=[]
        # Predict
        for batch_index, batch_samples in enumerate(test_loader):

            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            #data, target = batch_samples['img'], batch_samples['label']
            output = model(data)


            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()
            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)

    return targetlist, scorelist, predlist

### DenseNet

#class DenseNetModel(nn.Module):
#
#    def __init__(self):
#        """
#        Pass in parsed HyperOptArgumentParser to the model
#        :param hparams:
#        """
#        super(DenseNetModel, self).__init__()
#
#        self.dense_net = xrv.models.DenseNet(num_classes=2)
#        self.criterion = nn.CrossEntropyLoss()
#
#    def forward(self, x):
#        logits = self.dense_net(x)
#        return logits
#
#
#model = DenseNetModel().cuda()
#modelname = 'DenseNet_medical'


# In[146]:


# ### SimpleCNN
# class SimpleCNN(torch.nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__() # b, 3, 32, 32
#         layer1 = torch.nn.Sequential()
#         layer1.add_module('conv1', torch.nn.Conv2d(3, 32, 3, 1, padding=1))
#
#         #b, 32, 32, 32
#         layer1.add_module('relu1', torch.nn.ReLU(True))
#         layer1.add_module('pool1', torch.nn.MaxPool2d(2, 2)) # b, 32, 16, 16 //池化为16*16
#         self.layer1 = layer1
#         layer4 = torch.nn.Sequential()
#         layer4.add_module('fc1', torch.nn.Linear(401408, 2))
#         self.layer4 = layer4
#
#     def forward(self, x):
#         conv1 = self.layer1(x)
#         fc_input = conv1.view(conv1.size(0), -1)
#         fc_out = self.layer4(fc_input)
#
# model = SimpleCNN().cuda()
# modelname = 'SimpleCNN'
#
#
# # In[119]:
#
#
# ### ResNet18
# import torchvision.models as models
# model = models.resnet18(pretrained=True).cuda()
# modelname = 'ResNet18'
#
#
# In[106]:


### Dense121
import torchvision.models as models
model = models.densenet121(pretrained=True).cuda()
#model = models.densenet121(pretrained=True)
modelname = 'Dense121'


# # In[109]:
#
#
# ### Dense169
# import torchvision.models as models
# model = models.densenet169(pretrained=True).cuda()
# modelname = 'Dense169'
#
#
# # In[100]:
#
#
# ### ResNet50
# import torchvision.models as models
# model = models.resnet50(pretrained=True).cuda()
# modelname = 'ResNet50'
#
#
# # In[114]:
#
#
# ### VGGNet
# import torchvision.models as models
# model = models.vgg16(pretrained=True)
# model = model.cuda()
# modelname = 'vgg16'


# In[139]:


### efficientNet
#from efficientnet_pytorch import EfficientNet
#model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
#model = model.cuda()
#modelname = 'efficientNet-b0'


# model = EfficientNet.from_name('efficientnet-b1').cuda()
# modelname = 'efficientNet_random'


# train
bs = 10
votenum = 1
import warnings
warnings.filterwarnings('ignore')

r_list = []
p_list = []
acc_list = []
AUC_list = []
# TP = 0
# TN = 0
# FN = 0
# FP = 0
vote_pred = np.zeros(valset.__len__())
vote_score = np.zeros(valset.__len__())


#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
scheduler = StepLR(optimizer, step_size=1)

total_epoch = 15 # 3000
for epoch in range(1, total_epoch+1):
    train(optimizer, epoch)

    targetlist, scorelist, predlist = val(epoch)
    # print('target',targetlist)
    # print('score',scorelist)
    # print('predict',predlist)


    vote_pred = vote_pred + predlist
    vote_score = vote_score + scorelist

    print('Have save the trained model!!!')
    torch.save(model, "model_backup/{}.pt".format(modelname))

    if epoch % votenum == 0:

        # major vote
        vote_pred[vote_pred <= (votenum/2)] = 0
        vote_pred[vote_pred > (votenum/2)] = 1
        vote_score = vote_score/votenum

        print('training vote_pred', vote_pred)
        print('training targetlist', targetlist)
        TP = ((vote_pred == 1) & (targetlist == 1)).sum()
        TN = ((vote_pred == 0) & (targetlist == 0)).sum()
        FN = ((vote_pred == 0) & (targetlist == 1)).sum()
        FP = ((vote_pred == 1) & (targetlist == 0)).sum()


        # print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
        # print('TP+FP',TP+FP)
        p = TP / (TP + FP)
        # print('precision',p)
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        # print('recall',r)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        # print('F1',F1)
        print('acc',acc)
        AUC = roc_auc_score(targetlist, vote_score)
        # print('AUCp', roc_auc_score(targetlist, vote_pred))
        # print('AUC', AUC)



    # if epoch == total_epoch:
        # print('Have save the trained model!!!')
        # torch.save(model.state_dict(), "model_backup/{}.pt".format(modelname))

        vote_pred = np.zeros(valset.__len__())
        vote_score = np.zeros(valset.__len__())
        print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
        epoch, r, p, F1, acc, AUC))
        print('\n')

        f = open('model_result/{}.txt'.format(modelname), 'a+')
        f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
        epoch, r, p, F1, acc, AUC))
        f.close()


# In[145]:


# test
bs = 10
import warnings
warnings.filterwarnings('ignore')

r_list = []
p_list = []
acc_list = []
AUC_list = []
# TP = 0
# TN = 0
# FN = 0
# FP = 0
vote_pred = np.zeros(testset.__len__())
vote_score = np.zeros(testset.__len__())

#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
scheduler = StepLR(optimizer, step_size=1)

total_epoch = 3
for epoch in range(1, total_epoch+1):

    targetlist, scorelist, predlist = test(epoch)
    # print('target',targetlist)
    # print('score',scorelist)
    # print('predict',predlist)

    np.savetxt('target.txt', targetlist, delimiter=',')
    np.savetxt('predict.txt', predlist, delimiter=',')

    vote_pred = vote_pred + predlist
    vote_score = vote_score + scorelist

    TP = ((predlist == 1) & (targetlist == 1)).sum()
    TN = ((predlist == 0) & (targetlist == 0)).sum()
    FN = ((predlist == 0) & (targetlist == 1)).sum()
    FP = ((predlist == 1) & (targetlist == 0)).sum()

    # print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
    # print('TP+FP',TP+FP)
    p = TP / (TP + FP)
    # print('precision',p)
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    # print('recall',r)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    # print('F1',F1)
    print('acc',acc)
    # AUC = roc_auc_score(targetlist, vote_score)
    # print('AUC', AUC)

    if epoch % votenum == 0:

        # major vote
        vote_pred[vote_pred <= (votenum/2)] = 0
        vote_pred[vote_pred > (votenum/2)] = 1

#         print('vote_pred', vote_pred)
#         print('targetlist', targetlist)
        TP = ((vote_pred == 1) & (targetlist == 1)).sum()
        TN = ((vote_pred == 0) & (targetlist == 0)).sum()
        FN = ((vote_pred == 0) & (targetlist == 1)).sum()
        FP = ((vote_pred == 1) & (targetlist == 0)).sum()

        # print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
        # print('TP+FP',TP+FP)
        p = TP / (TP + FP)
        # print('precision',p)
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        # print('recall',r)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        # print('F1',F1)
        print('acc',acc)
        print('\n')
        # AUC = roc_auc_score(targetlist, vote_score)
        # print('AUC', AUC)


#         f = open('model_result/{modelname}.txt', 'a+')
#         f.write('precision, recall, F1, acc= \n')
#         f.writelines(str(p))
#         f.writelines('\n')
#         f.writelines(str(r))
#         f.writelines('\n')
#         f.writelines(str(F1))
#         f.writelines('\n')
#         f.writelines(str(acc))
#         f.writelines('\n')
#         f.close()


        vote_pred = np.zeros((1,testset.__len__()))
        vote_score = np.zeros(testset.__len__())
        print('testing vote_pred',vote_pred)
        # print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(epoch, r, p, F1, acc, AUC))
        print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}'.format(epoch, r, p, F1, acc))

        f = open('model_result/test_{}.txt'.format(modelname), 'a+')
        #f = open(f'model_result/test_{modelname}.txt', 'a+')
        # f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
        # epoch, r, p, F1, acc, AUC))
        f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}'.format(
        epoch, r, p, F1, acc))
        f.close()


# In[ ]:
