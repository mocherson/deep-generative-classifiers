from __future__ import print_function, division
from os.path import join
import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from PIL import Image
import time
from sklearn.metrics import roc_auc_score
import shutil

#path='/home/shared_data/chest_xray8/'

class ChestXray_Dataset(Dataset):
    """ChestXray dataset."""
    
    def __init__(self, path='/home/hddraid/shared_data/chest_xray8/', use='train', transform=None):
        """
        Args:
            csv_labelfile (string): Path to the csv file with labels.
            csv_bboxfile (string): Path to the csv file with bbox.
            root_dir (string): Directory with all the images.
            use (string): 'train' or 'validation' or 'test'
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.path=path
        csv_labelfile=join(path, 'Data_Entry_2017.csv')
        csv_bboxfile=join(path,'BBox_list_2017.csv')
        root_dir=join(path,'images/images')
        label_df = pd.read_csv(csv_labelfile)
        n=len(label_df)
        te = pd.read_csv(join(self.path,'test_list.txt'),header=None)[0]
        tr_val = pd.read_csv(join(self.path,'train_val_list.txt'),header=None)[0]
        tr, val = np.split(tr_val.sample(frac=1, random_state=0),[int(len(tr_val)*0.875),])
        
        if use=='train':
            self.label_df=label_df.loc[label_df['Image Index'].isin(tr)]
        elif use=='validation':
            self.label_df=label_df.loc[label_df['Image Index'].isin(val)]
        elif use=='test': 
            self.label_df=label_df.loc[label_df['Image Index'].isin(te)]
        elif use == "bboxtest":
            self.bbox=pd.read_csv(csv_bboxfile)
            #self.bbox['bbox']=self.bbox.iloc[:,[2,3,4,5]].apply(lambda x: tuple(x),axis=1)
            self.label_df=label_df.loc[label_df['Image Index'].isin( self.bbox['Image Index']), :]
        elif use == 'all':
            self.label_df = label_df
        else:
            raise Error('use must be "train" or "validation" or "test" or "bboxtest"')
            
        
        self.root_dir = root_dir
        self.classes = {'Atelectasis':0, 'Cardiomegaly':1, 'Effusion':2, 'Infiltration':3, \
                        'Mass':4, 'Nodule':5, 'Pneumonia':6, 'Pneumothorax':7, \
                        'Consolidation':8, 'Edema':9, 'Emphysema':10, 'Fibrosis':11, \
                        'Pleural_Thickening':12, 'Hernia':13   }
        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        img_name = self.label_df.iloc[idx, 0]
        image = Image.open(join(self.root_dir, img_name)).convert('RGB')
        labels = np.zeros(len(self.classes),dtype=np.float32)
        labels[[self.classes[x.strip()] for x in self.label_df.iloc[idx, 1].split('|') if x.strip() in self.classes]] = 1
        #bbox = self.box_loc.loc[self.box_loc['Image Index']==img_name,['Finding Label','bbox']] \
        #        .set_index('Finding Label').to_dict()['bbox']
        
        sample = {'image': image, 'label': labels, 'pid':self.label_df.iloc[idx, 3],  \
                  'age':self.label_df.iloc[idx, 4], 'gender':self.label_df.iloc[idx, 5], \
                  'position':self.label_df.iloc[idx, 6], 'name': img_name}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
    
class MyAlexNet(nn.Module):
    def __init__(self,outnum=14, gpsize=4):
        super(MyAlexNet, self).__init__()
        original_model = models.alexnet(pretrained=True)
        self.features = original_model.features
        self.features.add_module('transit',nn.Sequential(nn.Conv2d(256, 1024, 3, padding=1), nn.BatchNorm2d(1024),
                                                         nn.ReLU(inplace=True), nn.MaxPool2d(2,padding=1)))
        self.features.add_module('gpool',nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,1024)
        x = self.classifier(x)
        return x
    
    
class MyResNet50(nn.Module):
    def __init__(self,outnum=14,gpsize=4):
        super(MyResNet50, self).__init__()
        original_model = models.resnet50(pretrained=True) 

        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.features.add_module('transit',nn.Sequential(nn.Conv2d(2048, 1024, 3, padding=1), nn.BatchNorm2d(1024),
                                                         nn.ReLU(inplace=True), nn.MaxPool2d(2,padding=1)))
        self.features.add_module('gpool',nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,1024)
        x = self.classifier(x)
        return x
    
class MyVggNet16_bn(nn.Module):
    def __init__(self, outnum=14, gpsize=4):
        super(MyVggNet16_bn, self).__init__()
        original_model = models.vgg16_bn(pretrained=True)
        self.features = original_model.features
        self.features.add_module('transit',nn.Sequential(nn.Conv2d(512, 1024, 3, padding=1),nn.BatchNorm2d(1024),
                                                         nn.ReLU(inplace=True), nn.MaxPool2d(2,padding=1)))
        self.features.add_module('gpool',nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,1024)
        x = self.classifier(x)
        return x
    
class MyVggNet16(nn.Module):
    def __init__(self, outnum=14,gpsize=4):
        super(MyVggNet16, self).__init__()
        original_model = models.vgg16(pretrained=True)
        self.features = original_model.features
        self.features.add_module('transit',nn.Sequential(nn.Conv2d(512, 1024, 3, padding=1),nn.BatchNorm2d(1024),
                                                         nn.ReLU(inplace=True), 
                                                         nn.MaxPool2d(2,padding=1)))
        self.features.add_module('gpool',nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,1024)
        x = self.classifier(x)
        return x
    
class MyDensNet161(nn.Module):
    def __init__(self, outnum=14,gpsize=4):
        super(MyDensNet161, self).__init__()
        original_model = models.densenet161(pretrained=True)
        self.features = original_model.features
        self.features.add_module('transit',nn.Sequential(nn.Conv2d(2208, 1024, 3, padding=1), nn.BatchNorm2d(1024),
                                                         nn.ReLU(inplace=True), nn.MaxPool2d(2,padding=1)))
        self.features.add_module('gpool',nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,1024)
        x = self.classifier(x)
        return x
    
class MyDensNet201(nn.Module):
    def __init__(self, outnum=14, gpsize=4):
        super(MyDensNet201, self).__init__()
        original_model = models.densenet201(pretrained=True)
        self.features = original_model.features
        self.features.add_module('transit',nn.Sequential(nn.Conv2d(1920, 1024, 3, padding=1), nn.BatchNorm2d(1024),
                                                         nn.ReLU(inplace=True), nn.MaxPool2d(2,padding=1)))
        self.features.add_module('gpool',nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,1024)
        x = self.classifier(x)
        return x
    
class MyDensNet121(nn.Module):
    def __init__(self, outnum=14, gpsize=4):
        super(MyDensNet121, self).__init__()
        original_model = models.densenet121(pretrained=True)
        self.features = original_model.features
        self.features.add_module('transit',nn.Sequential(nn.Conv2d(1024, 1024, 3, padding=1), nn.BatchNorm2d(1024),
                                                         nn.ReLU(inplace=True), nn.MaxPool2d(2,padding=1)))
        self.features.add_module('gpool',nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,1024)
        x = self.classifier(x)
        return x
    
class VAE(nn.Module):
    def __init__(self, encoder, hid_channel=256, outnum=14):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.mu_fc =  nn.Sequential(nn.Conv2d(hid_channel, 1024, 3, padding=1), nn.BatchNorm2d(1024),   \
                                    nn.MaxPool2d(2,padding=1),nn.MaxPool2d(4))
        
        self.logv_fc = nn.Sequential(nn.Conv2d(hid_channel, 1024, 3, padding=1),    \
                                     nn.AvgPool2d(2,padding=1),nn.AvgPool2d(4))

        self.dec = nn.Linear(1024, outnum)
        
    def encode(self, x):
        h1 = self.encoder(x)
        return self.mu_fc(h1).view(-1,1024), self.logv_fc(h1).view(-1,1024)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
        else:
            z =  mu
        return z
    
    def decode(self, z):
        return self.dec(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
class VClassifier(nn.Module):
    def __init__(self, encoder, hid_channel=256, outnum=14):
        super(VClassifier, self).__init__()
        self.encoder = encoder
        self.mu_fc =  nn.Sequential(nn.Conv2d(hid_channel, 1024, 3, padding=1), nn.BatchNorm2d(1024),   \
                                    nn.MaxPool2d(2,padding=1),nn.MaxPool2d(4))
        
        self.logvar = nn.Parameter(torch.rand(1024))

        self.dec = nn.Linear(1024, outnum)
        
    def encode(self, x):
        h1 = self.encoder(x)
        return self.mu_fc(h1).view(-1,1024)
        
    def reparameterize(self, mu):
        if self.training:
            std = torch.exp(0.5*self.logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).expand_as(mu).add(mu)
        else:
            z =  mu
        return z
    
    def decode(self, z):
        return self.dec(z)
    
    def forward(self, x):
        mu = self.encode(x)
        z = self.reparameterize(mu)
        return self.decode(z), mu, self.logvar        
        

class W_BCEWithLogitsLoss(nn.Module):
    def __init__(self, size_average=False):
        super(W_BCEWithLogitsLoss, self).__init__()
        self.size_average = size_average
        
    def forward(self, input, target):
        p=int(target.sum().cpu().data.numpy())
        s=int(np.prod(target.size()))
        weight = target*(s/p-s/(s-p))+s/(s-p) if p!=0 else target+1
        return F.binary_cross_entropy_with_logits(input, target, weight, self.size_average)
    
class VLoss(nn.Module):
    def __init__(self, w=0):
        super(VLoss, self).__init__()
        self.reconloss = W_BCEWithLogitsLoss(size_average=False)
        self.w=w
        
    def forward(self, input, target, mu, logvar):
        BCE = self.reconloss(input, target)
        
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#         print('BCE=%s; KLD=%s'%(BCE,KLD))
        return BCE+ self.w*KLD

## train the CNN
def train(train_loader, model, criterion, optimizer, epoch, iter_size=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, data in enumerate(train_loader):
        # measure data loading time        
        data_time.update(time.time() - end)
        inputs, targets = Variable(data['image'].cuda()), Variable(data['label'].cuda())
        output = model(inputs)
        loss = criterion(output, targets)
        losses.update(loss.item(), inputs.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if (i+1) % iter_size == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
        
    return losses.avg
        
def train2(train_loader, model, criterion, optimizer, epoch, iter_size=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, data in enumerate(train_loader):
        # measure data loading time        
        data_time.update(time.time() - end)
        inputs, targets = Variable(data['image'].cuda()), Variable(data['label'].cuda())
        output, mu, logvar = model(inputs)
        loss = criterion(output, targets, mu, logvar)
        losses.update(loss.item(), inputs.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if (i+1) % iter_size == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
        
    return losses.avg

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        outputs=[]
        labels=[]
        for i, data in enumerate(val_loader):
            inputs, targets = data['image'].cuda(), data['label'].cuda()
            output = model(inputs)
            loss = criterion(output, targets)  
            losses.update(loss.item(),inputs.size(0))
            outputs.append(output)
            labels.append(targets)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % 20 == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses))
        outputs = torch.cat(outputs).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()
        
    roc = roc_auc_score(labels, outputs, average=None)
    avgroc = roc.mean()
    print('validate roc',roc)
    print('validate average roc',avgroc)
            
    return losses.avg, avgroc 

def validate2(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        outputs=[]
        labels=[]
        for i, data in enumerate(val_loader):
            inputs, targets = data['image'].cuda(), data['label'].cuda()
            output,mu, logvar = model(inputs)
            loss = criterion(output, targets, mu, logvar)  
            losses.update(loss.item(),inputs.size(0))
            outputs.append(output.cpu())
            labels.append(targets.cpu())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % 10 == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses))
        outputs = torch.cat(outputs).numpy()
        labels = torch.cat(labels).numpy()
        
    roc = roc_auc_score(labels, outputs, average=None)
    avgroc = roc.mean()
    print('validate roc',roc)
    print('validate average roc',avgroc)
            
    return losses.avg, avgroc        
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
                  
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('checkpoint','best'))

def test(test_loader, model):
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        outputs=[]
        labels=[]
        end = time.time()
        for i, data in enumerate(test_loader):
            inputs, targets = data['image'].cuda(), data['label']
            output = F.sigmoid(model(inputs))
            outputs.append(output)
            labels.append(targets)

            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()
            if (i+1) % 20 == 0:
                print("batch: [{}/{}], \t time:{}".format(i,len(test_loader),batch_time))

        outputs = torch.cat(outputs).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()
        
    
    roc = roc_auc_score(labels, outputs, average=None)
    avgroc = roc.mean()
    print('test roc',roc)
    print('test average roc',avgroc)
        
    return roc, avgroc, (labels,outputs)
        
def test2(test_loader, model):
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        outputs=[]
        labels=[]
        img = []
        end = time.time()
        for i, data in enumerate(test_loader):
            inputs, targets, name = data['image'].cuda(), data['label'], data['name']
            out = model(inputs)
            output = F.sigmoid(out[0] if isinstance(out, tuple) else out)
            outputs.append(output.cpu())
            labels.append(targets.cpu())
            img.append(name)

            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()
            if (i+1) % 20 == 0:
                print("batch: [{}/{}], \t time:{}".format(i,len(test_loader),batch_time))

        outputs = torch.cat(outputs).numpy()
        labels = torch.cat(labels).numpy()
        
    
    roc = roc_auc_score(labels, outputs, average=None)
    avgroc = roc.mean()
    print('test roc',roc)
    print('test average roc',avgroc)
        
    return roc, avgroc, (labels, outputs, img)


transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),  \
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                               ])


